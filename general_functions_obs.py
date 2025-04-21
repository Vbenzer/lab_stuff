from analysis.general_analysis import overplot_original_and_data

if __name__ == "__main__":
    #working_directory = "D:/Vincent/IFG_MM_0.3_TJK_2FC_PC_28_100_5/FF_with_all_filters"
    #get_ff_with_all_filters(working_directory)
    csv_file = "D:/Vincent/IFG_MM_0.3_TJK_2FC_PC_28_100_5/Throughput/Default Dataset.csv"
    measured_data = "D:/Vincent/IFG_MM_0.3_TJK_2FC_PC_28_100_5/Throughput/throughput.json"
    #inf_fiber_original_tp_plot(csv_file, show=True)
    overplot_original_and_data(measured_data, csv_file, show=True)