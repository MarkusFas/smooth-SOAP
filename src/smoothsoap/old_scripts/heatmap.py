    if ("heatmap" in plots) and len(methods_intervals) >= 2:
        interval_0 = methods_intervals[0]
        interval_1 = methods_intervals[1]
        cov1_int0 = interval_0[0].cov_mu_t
        cov2_int0 = interval_0[0].mean_cov_t
        cov1_int1 = interval_1[0].cov_mu_t
        cov2_int1 = interval_1[0].mean_cov_t
        for i, center in enumerate(interval_0[0].descriptor.centers):
            plot_heatmap(cov1_int0[i], cov1_int1[i], method.root + f'_temporal_interval{interval_0[0].interval}{interval_1[0].interval}_center{center}' + f'_{i}')
            plot_heatmap(cov2_int0[i], cov2_int1[i], method.root + f'_spatial_interval{interval_0[0].interval}{interval_1[0].interval}_center{center}' + f'_{i}')
        print('Plotted heatmap')