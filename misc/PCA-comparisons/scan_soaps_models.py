import numpy as np
import scipy as sp
import chemiscope
import featomic.torch
import ase.io
from tqdm import tqdm
#from metatensor.torch import Labels
from metatensor.torch import Labels, TensorBlock, TensorMap
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import itertools
import torch
from typing import Dict, List, Optional
from metatomic.torch import (
    AtomisticModel,
    ModelCapabilities,
    ModelEvaluationOptions,
    ModelMetadata,
    ModelOutput,
    System,
    systems_to_torch,
)


def get_trj(systems, alleve, atomsel, atomsel_elements, hypers, npca=40):
    #calculator = featomic.torch.SoapPowerSpectrum(**hypers)
    print('gettrj')
    print('atomsel',atomsel)
    print('atomsel_elements',atomsel_elements)
    print('systems',systems)
    model_soap,opts=setup_soap_predictor(atomsel,systems)
    #soap_pred = model_soap.forward(systems, options=opts, check_consistency=False)#, selected_keys=selection_O)
    trj_avg = np.zeros((len(systems),len(atomsel),npca) )
    for fidx, system in tqdm(enumerate(systems)):
        #rho2i = calculator.compute(system) # , selected_keys=selection)
        #rho2i = rho2i.keys_to_samples(["center_type"]).keys_to_properties(
        #    ["neighbor_1_type", "neighbor_2_type"]
        #)
        #new_soap = rho2i.block(0).values[atomsel]
        soap_pred = model_soap.forward([system], options=opts, check_consistency=False)#, selected_keys=selection_O)
        new_soap=soap_pred['features'][0].values
#        print('newsoap',new_soap.shape)
#        print('alleve',alleve.shape)
        for ielem,elem in enumerate(atomsel_elements):
#            print(elem)
            trj_avg[fidx,elem] =  np.einsum("ba,ib->ia", alleve[...,::-1][ielem,:,:npca], new_soap[elem]),
             #   np.einsum("ba,ib->ia", alleve[...,::-1][1,:,:npca], new_soap[ite])    
             #])
    return trj_avg

def setup_soap_predictor(atomsel,systems):
    print(atomsel)
    print(systems)
    print(torch.tensor(
            [
                [j, i]
                #for i in systems[0].types
                for i in atomsel
                for j in range(len(systems))
                # if j % 100 == 0
            ]
        ).shape)
    selected_atoms = Labels(
        ["system", "atom"],
        torch.tensor(
            [
                [j, i]
                #for i in systems[0].types
                for i in atomsel
                for j in range(len(systems))
                # if j % 100 == 0
            ]
        ),
    )
    #print(systems[0].types)
    #print(torch.unique(systems[0].types))
    species=torch.unique(systems[0].types).tolist()
    print(species)
    soap = SOAP_pred(species=species, hypers=hypers)
    soap.eval()    
    capabilities = ModelCapabilities(
        outputs={"features": ModelOutput(per_atom=True)},
        interaction_range=10.0,
        supported_devices=["cpu"],
        length_unit="A",
        atomic_types=species,
        dtype="float64",
    )
    
    metadata = ModelMetadata(name="SOAP water")
    model_soap = AtomisticModel(soap, metadata, capabilities)
    #model.save("soap_cv.pt", collect_extensions="extensions")
    
    #get soap calculated by model
    opts = ModelEvaluationOptions(
        length_unit="A",
        outputs={"features": ModelOutput(quantity="", per_atom=True)},
        selected_atoms=selected_atoms,
    )
    return model_soap, opts

def compute_autocorrelation_average(systems, hypers, kernel, atomsel, atomsel_element, maxlag=100):
    print('computeautocorrelation_average')
    print('atomsel',atomsel)
    print('atomsel_element',atomsel_element)
    model_soap,opts=setup_soap_predictor(atomsel,systems)
    soap_pred = model_soap.forward(systems, options=opts, check_consistency=False)#, selected_keys=selection_O)
    rho2i_values=soap_pred['features'][0].values
    #calculator = featomic.torch.SoapPowerSpectrum(**hypers)
    #rho2i_values=compute_soap_to_values(systems, calculator)
    
    buffer = np.zeros((len(atomsel), maxlag, rho2i_values.shape[1]))
    avgcov = np.zeros((len(atomsel_element), rho2i_values.shape[1], rho2i_values.shape[1],))
    soapsum = np.zeros((len(atomsel_element) ,rho2i_values.shape[1],))
    
    nsmp = np.zeros(len(atomsel))
    for fidx, system in tqdm(enumerate(systems)):
    
        #rho2i = calculator.compute(system) # , selected_keys=selection)
        #rho2i = rho2i.keys_to_samples(["center_type"]).keys_to_properties(
        #    ["neighbor_1_type", "neighbor_2_type"]
        #)
        #new_soap = rho2i.block(0).values[atomsel]
        soap_pred = model_soap.forward([system], options=opts, check_consistency=False)#, selected_keys=selection_O)
        new_soap=soap_pred['features'][0].values

        if fidx>=maxlag:
            first = buffer[:,fidx%maxlag]
            roll_kernel = np.roll(kernel, -fidx%maxlag)
            # computes a contribution to the correlation function
            # the buffer contains data from fidx-maxlag to fidx. add a forward ACF
            avg_soap = np.einsum("j,ija->ia", roll_kernel, buffer)
            for ielem,elem in enumerate(atomsel_element):
                 soapsum[ielem] += avg_soap[elem].sum(axis=0)
                 # ge and te averaged covariance
                 avgcov[ielem] += np.einsum("ia,ib->ab", new_soap[elem], new_soap[elem])
                 nsmp[ielem] += len(elem)
    
        buffer[:,fidx%maxlag] = new_soap

    avgcc=np.zeros((len(atomsel), new_soap.shape[1],new_soap.shape[1]))
    # autocorrelation matrix - remove mean
    for ielem,elem in enumerate(atomsel_element):
        avgcc[ielem] = avgcov[ielem]/nsmp[ielem] - soapsum[ielem, :, None]* soapsum[ielem, None, :]/(nsmp[ielem]**2)
    return avgcc

class SOAP_pred(torch.nn.Module):
    def __init__(self, species, hypers):
        super().__init__()

        self.neighbor_type_pairs = Labels(
            names=["neighbor_1_type", "neighbor_2_type"],
            values=torch.tensor(
                [[t1, t2] for t1 in species for t2 in species if t1 <= t2]
            ),
        )
        self.calculator = featomic.torch.SoapPowerSpectrum(**hypers)

    def forward(
        self,
        systems: List[System],
        outputs: Dict[str, ModelOutput],
        selected_atoms: Optional[Labels],
    ) -> Dict[str, TensorMap]:
        if "features" not in outputs:
            return {}

        if not outputs["features"].per_atom:
            raise ValueError("per_atom=False is not supported")

        soap = self.calculator(systems, selected_samples=selected_atoms)
        soap = soap.keys_to_samples("center_type")
        soap = soap.keys_to_properties(self.neighbor_type_pairs)
        return {"features": soap}


#Time correlations
stride = 1
npca=5
frameswater = ase.io.read("/Users/markusfasching/EPFL/Work/project1/scripts/SOAP-CV/traj_water_crop.xyz", ':')[::stride] #+ ase.io.read("", ":-500")[::stride]
framesice = ase.io.read("/Users/markusfasching/EPFL/Work/project1/scripts/SOAP-CV/traj_waterice.xyz", ':100')[::stride] #+ ase.io.read("", ":-500")[::stride]
frames = frameswater + framesice
for f in frames:
    f.wrap()
systems = systems_to_torch(frames, dtype=torch.float64)

test_frames=ase.io.read('/Users/markusfasching/EPFL/Work/project1/scripts/SOAP-CV/traj_ice_crop.xyz','-500:')[::stride]#+ase.io.read('../../mds/TET/positions.dump', '-500:')[::stride]
for f in test_frames:
    f.wrap()
test_systems = systems_to_torch(test_frames, dtype=torch.float64)
test_atomsel=[i.index for i in test_frames[0] if i.number==22][::2]
test_atomsel_element=[np.arange(len(test_atomsel))]
print('test_atomsel',test_atomsel)
print('test_atomsel_element',test_atomsel_element)

print('testlen',len(test_systems))

#atomsel=np.arange(0,len(frames[0]),16) # selects only a few atoms so it fits into memory
#ige = np.arange(len(atomsel)//2)
#ite = ige+(len(atomsel)//2)
atomsel=[i.index for i in frames[0] if i.number==22][::10]
atomsel=[i.index for i in frames[0]][::10]
atomsel=[i.index for i in frames[0]][::10]
print('atomsel',atomsel)

atomsel_element=[np.arange(len(atomsel))]
print('atomsel_element',atomsel_element)

maxlags=[1]#,5,10,50,100,200]#500]#,5,10,50,100,500,1000]
print('atomsel',atomsel)
cutoffs=[3]#,4,5]#,5,8]#[1,2,3,4,5,6,7,8,10]
angular_max=[1]#,2,4,6,8]
radial_max=[1]#,2,4,6,8]
combis=list(itertools.product(cutoffs, angular_max, radial_max, maxlags))
print('computing the following combis: ',combis)

for icombi,combi in enumerate(combis):
    print('now starting combi', combi)

    cutoff=combi[0]
    amax=combi[1]
    rmax=combi[2]
    maxlag=combi[3]

    # hypers for evaluating features
    hypers = {
        "cutoff": {"radius": cutoff, "smoothing": {"type": "ShiftedCosine", "width": 8}}, #8
        "density": {"type": "Gaussian", "width": 2.0}, 
        "basis": {
            "type": "TensorProduct",
            "max_angular": amax, #6
            "radial": {"type": "Gto", "max_radial": rmax}, #4
        },
    }
    
    #with averaging
    #postfix='av_{}'.format(maxlag)
    postfix='av_model_{}_{}_{}_{}'.format(cutoff,amax,rmax,maxlag)

    triangularkernel = 1-np.abs(np.arange(maxlag)-maxlag/2)*2/maxlag
    triangularkernel /= triangularkernel.sum()
    delta=np.zeros(maxlag)
    delta[maxlag//2]=1
    kernel=gaussian_filter(delta,sigma=maxlag//7)
    plt.plot(triangularkernel,'o-')
    plt.plot(delta,'o-')
    plt.plot(kernel, 'o-')
    plt.savefig('kernel_{}.png'.format(maxlag),dpi=100)
    #plt.show()
    plt.clf()
    
    
    #avgcc = avgcov/nsmp - soapsum[:, :, None]* soapsum[:, None, :]/(nsmp**2)
    avgcc= compute_autocorrelation_average(systems, hypers, kernel, atomsel, atomsel_element, maxlag=maxlag)
    plt.matshow(avgcc[0])
    plt.savefig(f'soap_{postfix}.png')
    plt.clf()
    
    #print('avgcc',avgcc.shape) 
    alleva, alleve = [None]*len(atomsel_element), [None]*len(atomsel_element)
    if len(atomsel_element)==1:
        alleva=[alleva]
        alleve=[alleve]
    for n in range(len(atomsel_element)):
        alleva[n], alleve[n] = sp.linalg.eigh(0.5*(avgcc[n]+avgcc[n].T))
    #print(alleva)
    alleva = np.array(alleva)
    alleve = np.array(alleve)

    torch.save(torch.tensor(alleva), 'alleva_{}.pt'.format(postfix))
    torch.save(torch.tensor(alleve), 'alleve_{}.pt'.format(postfix))

    alleva=torch.load('alleva_{}.pt'.format(postfix)).numpy()
    alleve=torch.load('alleve_{}.pt'.format(postfix)).numpy()

    
    for n in range(len(atomsel_element)):
        plt.loglog(alleva[n][::-1], '.-')
    plt.savefig(f'evas_loglog_{postfix}.png',dpi=100)
    plt.clf()
    for n in range(len(atomsel_element)):
        plt.plot(alleve[n][:,-1])
    plt.savefig(f'evas_{postfix}.png',dpi=100)
    plt.clf()
    
    
    trj_avg=get_trj(systems,alleve, atomsel, atomsel_element, hypers, npca=npca)
    print('pcashapeav',trj_avg)    
    for n in range(len(atomsel_element)):
        plt.plot(trj_avg[:,-1,n])
    #plt.plot(trj_avg[:,-1,1])
    plt.savefig(f'evas_{postfix}.png',dpi=100)
    plt.clf()
    plt.scatter(trj_avg[:,0,0], trj_avg[:,0,1], c=np.arange(len(trj_avg)), linestyle='-')
    plt.savefig(f'pca_{postfix}.png',dpi=100)
    plt.clf()
    print('trj',trj_avg.shape)    
    print('trj',trj_avg[:,0,0].shape)    
    
    # compute_autocorrelation_average(systems, hypers, kernel, atomsel, atomsel_element, maxlag=100)
    #model_soap,opts=setup_soap_predictor(test_atomsel,test_systems)
    #soap_pred = model_soap.forward(test_systems, options=opts, check_consistency=False)#, selected_keys=selection_O)
    #rho2i_values=soap_pred['features'][0].values
    #print('rho2i',rho2i_values.shape)
    print('alleve',alleve.shape)

    test_pca= get_trj(test_systems, alleve, test_atomsel, test_atomsel_element, hypers, npca=40)
    #test_pca=np.dot(rho2i_values,alleve[0].T)
    print('pcashape',test_pca.shape)
    test_pca=test_pca.reshape(test_pca.shape[0]*test_pca.shape[1],test_pca.shape[2])
    print('pcashape',test_pca.shape)

    plt.plot(test_pca[:,0],test_pca[:,1],'o')
    plt.savefig('test_pca_{}.png'.format(postfix),dpi=100)
    plt.clf()
