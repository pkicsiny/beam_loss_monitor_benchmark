import pandas as pd, numpy as np, matplotlib.dates as dates, seaborn as sns, paramiko, os, re, sys, tables, glob, sys, shutil
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from ipywidgets import interact, IntSlider


"""
Some parts of this code are credited to V. Okhotnikov.
"""


# conversion factor from bits to dose [Gy/bits]
α = 3.62e-9

# conversion factor from bits to current [bits/A]
β = 2.048e8

# no. of 40 μs steps for each running sum, running sum data divided by this yields the data per 0.04 ms (running avg.)
γ = [1, 2, 8, 16, 64, 256, 2048, 16384, 32768, 131072, 524288, 2097152]

# RS integration times in [s]
T = [4e-5 * γi for γi in γ]

DETECTOR_NAMES = {1: ('BCML1 +Z near top AB', 'CMS08', 0.855, 40, 55, 424, 500, 'deeppink', '-'),
                  2: ('BCML1 +Z near bottom AB', 'P34', 0.855, 40, 100, 483, 200, 'deeppink', '--'),
                  3: ('BCML1 +Z far top AB', 'CMS12', 0.855, 40, 100, 509, 500, 'deeppink', '-.'),
                  4: ('BCML1 +Z far bottom AB', 'CMS10', 0.855, 40, 100, 500, 500, 'deeppink', ':'),
                  5: ('BCML1 -Z near top AB', 'P36', 0.855, 40, 100, 512, 500, 'deepskyblue', '-'),
                  6: ('BCML1 -Z near bottom AB', 'CMS06', 0.855, 40, 55, 419, 500, 'deepskyblue', '--'),
                  7: ('BCML1 -Z far top AB', 'CMS05', 0.855, 40, 100, 520, 500, 'deepskyblue', '-.'),
                  8: ('BCML1 -Z far bottom AB', 'CMS07', 0.855, 40, 100, 493, 500, 'deepskyblue', ':'),

                  9: ('empty', 'empty', 0, 0, 0, 0, 0, 'none', '-'),
                  10: ('empty', 'empty', 0, 0, 0, 0, 0, 'none', '-'),
                  11: ('empty', 'empty', 0, 0, 0, 0, 0, 'none', '-'),
                  12: ('empty', 'empty', 0, 0, 0, 0, 0, 'none', '-'),
                  13: ('empty', 'empty', 0, 0, 0, 0, 0, 'none', '-'),
                  14: ('empty', 'empty', 0, 0, 0, 0, 0, 'none', '-'),
                  15: ('empty', 'empty', 0, 0, 0, 0, 0, 'none', '-'),
                  16: ('empty', 'empty', 0, 0, 0, 0, 0, 'none', '-'),

                  17: ('BCML2 -Z far 1 AB', 'CMS14 (Pos)', 0.855, 40, 55, 409, 400, 'blue', '-'),
                  18: ('BCML2 -Z far 2', 'Sap_Stack (6-7)', 0.855, 40, 55, 414, 500, 'blue', '--'),
                  19: ('BCML2 -Z far 3 AB', 'P39', 0.855, 40, 20, 400, 200, 'lightseagreen', '-'),
                  20: ('BCML2 -Z far 4', '3D-SmallPitch', 0.855, 40, 20, 416, 0, 'navy', '-'),
                  21: ('BCML2 -Z far 5', 'Sapphire (10)', 0.855, 40, 20, 415, 500, 'lightseagreen', '-.'),
                  22: ('BCML2 -Z far 6', 'Sap_Stack (8-9)', 0.855, 40, 20, 425, 500, 'lightseagreen', '--'),
                  23: ('Empty -Z far 7', 'Empty', 0.855, 0, 0, 500, 0, 'yellow', '-'),
                  24: ('BLM tube -Z', 'BLM tube -Z', 0.855, 0, 0, 4740, 0, 'lime', ':'),

                  25: ('BCML2 -Z near 1 AB', 'CMS15 (Neg)', 0.855, 40, 40, 421, 400, 'blue', '-.'),
                  26: ('BCML2 -Z near 2', 'P03', 0.855, 40, 40, 400, 200, 'blue', ':'),
                  27: ('BCML2 -Z near 3 AB', 'P13', 0.855, 40, 20, 418, 160, 'lightseagreen', ':'),
                  28: ('BCML2 -Z near 4', 'DOI_C (Neg)', 0.855, 40, 20, 414, 0, 'lightseagreen', '-'),
                  29: ('BCML2 -Z near 5', 'Empty', 0.855, 40, 20, 411, 0, 'lightseagreen', '--'),
                  30: ('BCML2 -Z near 6', 'Empty', 0.855, 40, 20, 410, 0, 'navy', '--'),
                  31: ('BCML2 -Z near 7', 'Empty', 0, 0, 0, 0, 0, 'none', '-'),
                  32: ('BCML2 -Z near 8', 'Empty', 0, 0, 0, 0, 0, 'none', '-'),

                  33: ('BCML2 +Z near 1 AB', 'CMS12 (Neg)', 0.855, 40, 30, 414, 350, 'indigo', '-'),
                  34: ('BCML2 +Z near 2', 'Sap_Stack (1-5)', 0.855, 40, 65, 413, 500, 'red', '-'),
                  35: ('BCML2 +Z near 3 AB', 'P47', 0.855, 40, 40, 407, 200, 'red', '--'),
                  36: ('BCML2 +Z near 4', '3D-BigPitch', 0.855, 40, 30, 418, 45, 'brown', '-'),
                  37: ('BCML2 +Z near 5', 'Empty', 0.855, 40, 30, 413, 0, 'indigo', '-'),
                  38: ('BCML2 +Z near 6', 'Empty', 0.855, 40, 40, 414, 0, 'indigo', '--'),
                  39: ('BCML2 +Z near 7', 'Empty', 0.855, 7.5, 100, 500, 0, 'lime', '-'),
                  40: ('BCML2 +Z near 8', 'Empty', 0.18, 90, 100, 491, 0, 'lime', '-.'),

                  41: ('BCML2 +Z far 1 AB', 'CMS13 (neg)', 0.855, 40, 30, 423, 350, 'brown', '--'),
                  42: ('BCML2 +Z far 2', 'P02 (neg)', 0.855, 40, 100, 473, 200, 'red', '-.'),
                  43: ('BCML2 +Z far 3 AB', 'P44', 0.855, 40, 40, 414, 500, 'indigo', '-.'),
                  44: ('BCML2 +Z far 4', 'DOI_01 (neg)', 0.855, 40, 40, 405, 500, 'red', ':'),
                  45: ('BCML2 +Z far 5', 'Empty', 0.855, 7.5, 100, 500, 0, 'lime', '--'),
                  46: ('BCML2 +Z far 6', 'Empty', 0.855, 45, 40, 410, 0, 'indigo', ':'),
                  47: ('BCML2 +Z far 7', 'Empty', 0.855, 0, 0, 500, 0, 'y', '-'),
                  48: ('BLM tube +Z', 'BLM tube +Z', 0.855, 0, 0, 4740, 0, 'lime', ':')}


def freeze_header(df, num_rows=30, num_columns=12, step_rows=1, step_columns=1):
    """
    idea: https://stackoverflow.com/questions/28778668/freeze-header-in-pandas-dataframe
    Freeze the headers (column and index names) of a Pandas DataFrame. A widget
    enables to slide through the rows and columns.
    Parameters
    ----------
    df : Pandas DataFrame
        DataFrame to display
    num_rows : int, optional
        Number of rows to display
    num_columns : int, optional
        Number of columns to display
    step_rows : int, optional
        Step in the rows
    step_columns : int, optional
        Step in the columns
    Returns
    -------
    Displays the DataFrame with the widget
    """

    @interact(last_row=IntSlider(min=min(num_rows, df.shape[0]),
                                 max=df.shape[0],
                                 step=step_rows,
                                 description='rows',
                                 readout=False,
                                 disabled=False,
                                 continuous_update=True,
                                 orientation='horizontal',
                                 slider_color='purple'),
              last_column=IntSlider(min=min(num_columns, df.shape[1]),
                                    max=df.shape[1],
                                    step=step_columns,
                                    description='columns',
                                    readout=False,
                                    disabled=False,
                                    continuous_update=True,
                                    orientation='horizontal',
                                    slider_color='purple'))

    def _freeze_header(last_row, last_column):
        display(df.iloc[max(0, last_row - num_rows):last_row,
                max(0, last_column - num_columns):last_column])


def get_column_names(channels, rss):
    return ['CH' + str(channel) + 'RS' + str(rs) for channel in channels for rs in rss]


def read_supertable_csv(csv_path, read_cols=["fill", "cms_delivered"], sort_by="fill", dropna=True):
    """
        Reads superable data from https://acc-stats.web.cern.ch/acc-stats/#lhc/super-table
        that was exported to .csv file.
        :param csv_path: string of path ending with the .csv file name e.g. path/to/file.csv
        :param read_cols: list of strings containing column names of the supertable to read.
                         Reads cms delivered lumi data by default. If [] (empty list) is given,
                          reads all columns.
        :param sort_by: string, sorts dataframe by this parameter. Default is sorting by fill.
        :param dropna: bool, drop rows with any NaN entry if True. Default is True.

        :return selected_stats: pandas dataframe of selected columns of supertable
    """

    stats_supertable = pd.read_csv(csv_path)
    selected_stats = stats_supertable[read_cols] if len(read_cols) > 0 else stats_supertable

    # drop invalid lumi fills
    if dropna:
        selected_stats.dropna(inplace=True)
    selected_stats = selected_stats.loc[(selected_stats != 0).all(1)]

    # sort by fill number
    if sort_by in selected_stats.columns:
        selected_stats = selected_stats.sort_values(sort_by).reset_index(drop=True)
    return selected_stats


def read_hd5(hd5_list, drive):
    """
        Reads BCML data from .hd5 files into a pandas dataframe.
        :param hd5_list: list of strings, containing hd5 file names, e.g. [file1.hd5, file2.hd5, ...]
        :param drive: string, path to hd5 files

        :return result_df: pandas dataframe containing the data from the hd5 files. Column names are
                           BCML readout channels, row indices are timestamps.
        :return fillnum_df: pandas dataframe containing the fill number for each timestamp.
                            Row indices are the same timestamps as in *fillnum_df*. Has same size as *result_df*
        :return fillnum_list: list of unique fill numbers in the loaded data
    """

    # typecheck
    if not isinstance(hd5_list, list):
        raise TypeError("List of hd5 files must have a list type.")

    # output
    result_df = pd.DataFrame()
    fillnum_df = pd.DataFrame()
    fillnum_list = []

    # loop over hd5 files
    total_hd5 = len(hd5_list)
    for idx, filename in enumerate(hd5_list):
        print("\nReading file [{}/{}]:".format(idx + 1, total_hd5), os.path.join(drive, filename))
        hd5file = tables.open_file(filename)
        try:

            # no. of data entries = no. of timestamps, each with an acquired data value for all 576 (48*12) channels
            raw_data = hd5file.root.bcm

            # see if we can access the data in the file (not always successful)
            fillnum_data = raw_data[:]['fillnum']
            fillnum_set = list(set(fillnum_data))
            print("Available fills in file:", fillnum_set)

            # print time interval of file
            acq_time = pd.to_datetime(raw_data[:]['acqts'][:, 0], unit='s')
            print("Time range of data in file:", min(acq_time), "-", max(acq_time))
        except:
            print("Could not open file. Moving on to the next file.")
            hd5file.close()
            continue

        # log unique fillnums
        [(fn not in fillnum_list and fillnum_list.append(fn)) for fn in fillnum_set]

        # make dataframe
        result_df_file = pd.DataFrame(raw_data[:]['acq'], index=acq_time,
                                      columns=get_column_names(range(1, 49), range(1, 13)))
        print("Size of dataframe: ", np.shape(result_df_file))
        fillnum_df_file = pd.DataFrame(fillnum_data, index=acq_time, columns=["fill"])

        # concatenate to result
        result_df = pd.concat([result_df, result_df_file])
        fillnum_df = pd.concat([fillnum_df, fillnum_df_file])

        # important to close file otherwise old data is stuck in memory!!!
        hd5file.close() if hd5file.isopen else print("File closed.")

    # sort by date + assertions
    fillnum_list = sorted(fillnum_list)
    assert len(fillnum_list) == len(fillnum_df["fill"].unique())
    assert len(result_df) == len(fillnum_df)
    return result_df, fillnum_df, fillnum_list


def plot_fill_data(data_df, fillnum_df, fills_to_plot, channels_to_plot, rs=12, logy=True, out_dir="blm_fill_images"):
    """
        Utility function to plot hd5 data on a per fill basis.
        :param data_df: pandas dataframe containing the data from the hd5 files. Column names are
                        BCML readout channels, row indices are timestamps.
        :param fillnum_df: pandas dataframe containing the fill number for each timestamp.
                        Has same size as *data_df*
        :param fills_to_plot: list of unique fill numbers to plot
        :param channels_to_plot: list of BCM channels (1-48) to plot
        :param rs: int, running sum to plot
        :param logy: bool, use logarithmic y axis on plots
        :param out_dir: string, plot save folder
    """


    # make output directory
    os.makedirs(out_dir, exist_ok=True)

    # Formatting x-axis - format UTC-time: Hours:Minutes
    hours = dates.HourLocator(interval=2)
    h_fmt = dates.DateFormatter('%m/%d/%H:%M')

    for fn in fills_to_plot:
        print("Plotting fill {}".format(fn))
        fn_df = data_df[fillnum_df["fill"].isin([fn])]
        ch_tag = "{}-{}".format(min(channels_to_plot), max(channels_to_plot)) if len(
            channels_to_plot) > 4 else channels_to_plot
        fill_tag = fn

        # making plot
        fig, ax = plt.subplots(figsize=(20, 6))
        plt.grid()

        #loop over selected channels
        for ch in channels_to_plot:

            # one column of dataframe
            ch_fn_df = fn_df['CH' + str(ch) + 'RS' + str(rs)]
            ax.plot(ch_fn_df.index, ch_fn_df.values, label="CH{}: {}".format(ch, DETECTOR_NAMES[ch][0]), linewidth=1)

            # Plot formatting
            ax.xaxis.set_major_locator(hours)
            ax.xaxis.set_major_formatter(h_fmt)
            plt.yticks(fontsize=13)
            plt.xticks(fontsize=13)
            ax.set_title('Fill {}, CH {}, RS {}'.format(fill_tag, ch_tag, rs), fontsize=19)
            ax.set_ylabel('Signal current (A)', fontsize=19)
            ax.set_xlabel('Time (UTC)', fontsize=19)
            plt.legend(loc="best", prop={'size': 13})
            if logy:
                ax.set_yscale("log")
            plt.savefig(out_dir + "/data_fill_{}".format(fn))
        plt.close()


def sim_data_ratios_per_month(excel_ratio_files, excel_error_files, excel_path, colnames=get_column_names([24, 48], [10]),
                              column_for_index=None, outliers = [0.5, 1.5]):
    """
        Aggregates FLUKA/measurement data ratios from per pill values (input excels) to monthly values.
        :param excel_ratio_files: lis of strings, excel file names to read
        :param excel_error_files: lis of strings, excel file names to read containing stat uncertainties
        :param excel_path: string, path to excel files
        :param colnames: list of column names to give to the dataframes
        :param column_for_index: int or string, column index or name to use as index in the
        result dataframes (usually fill numbers)
        :param outliers: list of outlier thresholds. Rows with ratios outside the interval will be dropped.
        Returns don't contain outliers.

        :return sim_data_ratio_dict: dict of pandas dataframes containing sim/data ratios per fill. Keys are months.
        :return sim_data_ratio_df: pandas dataframe containing sim/data ratios per fill, concatted from all excel files
        (usually per year).
        :return sim_data_ratio_monthly_df: pandas dataframe of monthly mean and std. dev. sim/data ratios
    """

    # contains data for each month, per fill
    ratios_dict = {}
    errors_dict = {}

    # contains data from all excel files, per fill
    yearly_ratios_df = pd.DataFrame()
    yearly_errors_df = pd.DataFrame()

    # contains data from all excel files, averaged per month
    monthly_ratios_df = pd.DataFrame()
    monthly_errors_df = pd.DataFrame()

    # loop over monthly excels files
    for (ratios, errors) in zip(sorted(excel_ratio_files), sorted(excel_error_files)):
        try:
            file = pd.read_excel(os.path.join(excel_path, ratios))
            ratios_df = file[colnames]

            # read uncertainty excel per fill
            errors_file = pd.read_excel(os.path.join(excel_path, errors))
            errors_df = errors_file[colnames]

            # try using a column for indices (should contain fill numbers)
            if isinstance(column_for_index, int):
                ratios_indices = file.iloc[:, column_for_index]
                errors_indices = errors_file.iloc[:, column_for_index]
                print("Using column {} for index.".format(column_for_index))
            elif isinstance(column_for_index, str):
                ratios_indices = file[column_for_index]
                errors_indices = errors_file[column_for_index]
                print("Using column {} for index.".format(column_for_index))
            else:
                ratios_indices = file.index
                errors_indices = errors_file.index
                print("Using default index.")
            ratios_df.index = ratios_indices
            errors_df.index = errors_indices
        except:
            raise RuntimeError("Invalid column names.")

        # month tag e.g. "05" (files must be named like e.g. anything_MM.xlsx)
        month = re.findall("[0-9]+\.", ratios)[0][:-1]
        print(month)

        # filter outliers (where the data is nothing but baseline but still there is delivered lumi value)
        #errors_df = errors_df[(ratios_df.min(axis=1) > outliers[0]) & (ratios_df.max(axis=1) < outliers[1])]
        #ratios_df = ratios_df[(ratios_df.min(axis=1) > outliers[0]) & (ratios_df.max(axis=1) < outliers[1])]

        # add monthly dataframe to dict
        ratios_dict[month] = ratios_df
        errors_dict[month] = errors_df

        # append to annual dataframe
        yearly_ratios_df = yearly_ratios_df.append(ratios_df)
        yearly_errors_df = yearly_errors_df.append(errors_df)

        # calculate mean and std of monthly data
        monthly_ratios_df = monthly_ratios_df.append(
            pd.concat([pd.DataFrame([round(ratios_df.mean(), 3),
                                     round(ratios_df.std(), 3)],
                                    index=["mean", "σ"]).T], keys=[month]))
        monthly_errors_df = monthly_errors_df.append(
            pd.concat([pd.DataFrame([round(np.sqrt((errors_df ** 2).sum()) / len(errors_df), 6)],
                                    index=["stat unc. of mean of ratios due to baseline noise"]).T], keys=[month]))

    # ratios per fill in dict, ratios per fill in one df, ratios per month in one df
    return ratios_dict, yearly_ratios_df, monthly_ratios_df, errors_dict, yearly_errors_df, monthly_errors_df


def plot_blm_data(df, rs=9, hours=dates.HourLocator(interval=48), h_fmt=dates.DateFormatter('%m/%d'), save=False,
                  fill=None):
    fig, ax = plt.subplots(figsize=(20, 6))
    plt.grid()
    labels = {24: "BLM -Z", 48: "BLM +Z"}

    for ch in [24, 48]:
        # one column of dataframe
        ch_df = df['CH' + str(ch) + 'RS' + str(rs)]
        x = [pd.Timestamp(t) for t in ch_df.index.values]
        ax.plot(x, ch_df.values, label=labels[ch], linewidth=1)

        # Plot formatting
        ax.xaxis.set_major_locator(hours)
        ax.xaxis.set_major_formatter(h_fmt)
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)

    ax.set_ylabel('Signal current [A]', fontsize=19)
    ax.set_xlabel('Time [UTC]', fontsize=19)
    ax.set_yscale("log")
    plt.legend(loc="lower right", prop={'size': 14})

    # set up title and plot dir
    year = str(df.index[0])[:4]
    month = str(df.index[0])[5:7]
    #title = '{}/{}, CH [24, 48], RS 10'.format(year, month)
    plot_dir = "../json_plots/{}_per_month/{}-{}_rs10.png".format(year, year, month)
    if fill is not None:
        title = 'Fill {}, '.format(fill) + title
        plot_dir = "../json_plots/{}_per_fill/{}_{}/{}-{}_fill_{}_rs10.png".format(year, year, month, year, month, fill)
    #ax.set_title(title, fontsize=19)
    if save:
        os.makedirs(re.findall(".*/", plot_dir)[0], exist_ok=True)
        plt.savefig(plot_dir)
    plt.close()


def select_lumi_data(lumi_df, df, fill_key="fill", normtag_key="cms_delivered", lumi_threshold=0):
    """
    Returns lumi data for fills that are present in *df* dataframe.
    :param lumi_df: dataframe of CMS delivered lumi values
    :param df: dataframe of BLM measurement data
    :param fill_key: string for column name that contains fill numbers
    :param normtag_key: string for column name that contains delivered luminosity data
    :param lumi_threshold: float for rejecting fills where the total delivered lumi does not reach this value
    :return: subdataframe of *lumi_df* only with those fills that are present in *df*
    """

    return lumi_df[(lumi_df[fill_key].isin(df[fill_key])) & (lumi_df[normtag_key] > lumi_threshold)].reset_index(drop=True)


def calculate_sim_data_ratios(sim_df, data_df, column_names, fill_key="fill"):
    """
    Returns dataframe with simulation/measurement data ratios. Simulation data per fill should be in *lumi_df* in columns
    *sim_charge_-Z* and *sim_charge_+Z*. Measured total charge per fill should be in *df* under the columns *column_names*
    (-Z, +Z respectively).
    :param sim_df: dataframe of CMS delivered lumi values, Rpp factors and simulated charge sums
    :param data_df: dataframe of BLM measurement data
    :param column_names: list of column names for +/- Z BLM tubes
    :param fill_key: string for column name that contains fill numbers

    :return: dataframe containing fill numbers and the +/-Z BLM sim/data ratios
    """

    return pd.DataFrame([(
        (sim_df["sim_charge_-Z"][i] / data_df.loc[sim_df[fill_key][i]])[column_names[0]],
        (sim_df["sim_charge_+Z"][i] / data_df.loc[sim_df[fill_key][i]])[column_names[1]]
    ) for i in range(len(sim_df))],
        index=sim_df[fill_key],
        columns=column_names)


def sim_data_ratio_plot(sim_data_ratio_df, year, show_mean=False):
    """
    :param sim_data_ratio_df: dataframe containing simulation/measurement fractions for +/- Z BLM tubes.
    Indices are fill numbes.
    :param year: int, e.g. 2018
    :param median: plot median value
    :return: the figure
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.grid()
    plt.yticks(fontsize=13)
    plt.xticks(fontsize=13)

    # assumes -Z +Z order in dataframe
    labels = ["BLM -Z", "BLM +Z"]
    ax.scatter(sim_data_ratio_df.index, sim_data_ratio_df.iloc[:, 0], c="b", label=labels[0])
    ax.scatter(sim_data_ratio_df.index, sim_data_ratio_df.iloc[:, 1], c="r", label=labels[1])
    ax.set_ylabel('FLUKA/measurement ratio', fontsize=19)
    ax.set_xlabel('Proton-proton fills', fontsize=19)
    ax.legend(fontsize=14)
    ax.set_ylim(0.4, 1.2)
    ax.get_xaxis().set_label_coords(0.86,-0.075)
    ax.get_yaxis().set_label_coords(-0.07,0.7)
    ax.set_title(r'$\bf{{{CMS}}}$ offline luminosity 2018 ($\sqrt{\mathrm{s}}$=13 TeV)', loc='right', fontsize=16)

    # plot mean
    if show_mean:
        handle = fig.gca()
        y24_tot = list(zip(*np.concatenate([sub.get_offsets() for sub in handle.collections[::2]])))[1]
        y48_tot = list(zip(*np.concatenate([sub.get_offsets() for sub in handle.collections[1::2]])))[1]
        plt.axhline(y=np.mean(y24_tot), color='b', linestyle='-', label=str(round(np.mean(y24_tot), 3))+r"$\pm$"
            +str(round(np.std(y24_tot), 3)))
        plt.axhline(y=np.mean(y48_tot), color='r', linestyle='-', label=str(round(np.mean(y48_tot), 3))+r"$\pm$"
            +str(round(np.std(y48_tot), 3)))

        # order legends
        hndls, lbls = ax.get_legend_handles_labels()
        hndls = [hndls[3], hndls[1], hndls[2], hndls[0]]
        lbls = [lbls[3], lbls[1], lbls[2], lbls[0]]
        ax.legend(hndls, lbls, fontsize=12)

    else:
        ax.legend(fontsize=12)
        
    plt.close()
    return fig
