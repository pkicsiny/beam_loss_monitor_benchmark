import numpy as np, matplotlib.dates as dates, os, sys, pickle
from matplotlib import pyplot as plt
import json
import time, datetime


"""
global variables
"""


#path of json file
json_path = "webmonitor-es-bril-dipanalyzermon_old2018.json"
#json_path = "C:/Users/pkicsiny/Desktop/TSC_CERN/BLM_study/json_data/webmonitor-es-bril-dipanalyzermon_old2018-short.json"

#set up output directories
out_dir = "json_pickles"
if not os.path.exists(out_dir):
	os.makedirs(out_dir)

plot_dir = "json_plots"
if not os.path.exists(plot_dir):
	os.makedirs(plot_dir)

#set up output dfs
months = [str(m).zfill(2) for m in list(range(4,11))]
cols = ["CH24RS9", "CH48RS9"]

#set up month intervals
start_dates = ["2018-"+m+"-01T00:00:00Z" for m in months]
end_dates = ["2018-"+str(int(m)+1).zfill(2)+"-01T00:00:00Z" for m in months]

#json read parameters
block_size = 300000
plot_freq = 3000


"""
functions
"""


def get_json_data(json_line, result_dict):
    ts = json_line['_source']['timestamp']
    for m, start_date, end_date in zip(months, start_dates, end_dates):
        if datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ") >= datetime.datetime.strptime(start_date, "%Y-%m-%dT%H:%M:%SZ")\
	and datetime.datetime.strptime(ts, "%Y-%m-%dT%H:%M:%SZ") < datetime.datetime.strptime(end_date, "%Y-%m-%dT%H:%M:%SZ"):
             result_dict[m].append((ts, json_line["_source"]["RunningSum9"][23],  json_line["_source"]["RunningSum9"][47]))

                
def read_block(file_handler, block_size=200):
    block = []
    restart_timer = True
    for line in file_handler:
        if restart_timer:
            start = time.time()
            restart_timer = False
        block.append(line)
        if len(block) == block_size:
            end = time.time()
            restart_timer = True
	    dt = end - start
            print("Block loaded in "+str(dt)+" seconds")
            yield block
            block = []
    if block:
        end = time.time()
	dt = end - start
        print("Block loaded in "+str(dt)+" seconds")
        yield block
        

def plot_read_progress(ax, start, idx, current_block_idx, plot_dir):
    loop = time.time()
    if not os.path.exists(plot_dir):
	os.makedirs(plot_dir)
    ax.scatter(idx, loop - start, c="b")
    ax.set_xlabel("json row index", fontsize=19)
    ax.set_ylabel("time elapsed [s]", fontsize=19)
    cb = current_block_idx + 1
    plt.title("json block # "+str(cb), fontsize=20)
    plt.savefig(plot_dir+"/json_block_"+str(cb)+".png")


"""
read json
"""


with open(json_path) as file_handler:

#get data by blocks, this has some additional latency
    for current_block_idx, block in enumerate(read_block(file_handler, block_size=block_size)):
        
#set up break condition
        if current_block_idx == 9999:
            break
	cb = current_block_idx + 1
        print("Reading block: # "+str(cb))

#each month is a list of tuples with (timestamp, ch24, ch48)
	result_dict = {}
	for m in months:
        	result_dict[m] = []

#set up plot for block data
        ax = plt.subplot(111)
        plt.grid()
        plt.yticks(fontsize=13)
        plt.xticks(fontsize=13)

#start time counter
        start = time.time()

#loop over lines in block
        for idx in range(len(block)):
        
#read single line
            json_line = json.loads(block[idx])
    
#append row data to proper dataframe
            get_json_data(json_line, result_dict)

#make plot on performance
            if idx%plot_freq == 0:
                plot_read_progress(ax, start, idx, current_block_idx, plot_dir)
            
#save data at each block limit
        print("Saving block # "+str(cb))
        for m in months:
            with open(out_dir+"/month_"+m+"_batch_"+str(cb), 'wb') as handle:
                pickle.dump(result_dict[m], handle, protocol=pickle.HIGHEST_PROTOCOL)

#close file and plot
        plt.close()
    file_handler.close()
