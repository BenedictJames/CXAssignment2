import pandas as pd
from gurobipy import *

# Importing the excel file
RCPSP_data = pd.ExcelFile('RCPSP_data.xlsx')
# Extracting excel sheet names
sName = RCPSP_data.sheet_names
# Creating dictionary with key = sheet name, value = df
RCPSP_data_dict = RCPSP_data.parse(sName)
# Create df with results excel sheet
Results_df = RCPSP_data_dict[sName[0]]
# Create dictionary with test instances: with key = sheet name, value = df
RCPSP_inst_dict = RCPSP_data.parse(sName[1:])


# Set col names of df's in RCPSP_inst_dict to col_names
# Create dictionary with keys = original column names  and values = new column names
orig_col_names = list(RCPSP_data_dict[sName[1]].columns)
new_col_names = ["d_i", "r_i_1", "r_i_2", "r_i_3", "r_i_4", "s_i", "j_1_i", "j_2_i", "j_3_i"]
names_dict = {orig_col_names[i]: new_col_names[i] for i in range(len(orig_col_names))}
# function for changing column names of df
def rename_columns(df, dict):
    return df.rename(columns = dict)
# Apply function to dictionary containing all RCPSP test instances as dfs; this will name the dfs' columns
RCPSP_inst_dict_named = {k: rename_columns(v, names_dict) for k,v in RCPSP_inst_dict.items()}

# Function for extracting the single test instance x df from dictionary (df)
def df_inst(x):
    return RCPSP_inst_dict_named[sName[x]]
# Function for extracting n for test instance x from dictionary (int)
def n_inst(x):
    return int(RCPSP_inst_dict_named[sName[x]].iloc[0, 0])
# Function for extracting k for test instance x from dictionary (int)
def k_inst(x):
    return int(RCPSP_inst_dict_named[sName[x]].iloc[0, 1])
# Function for extracting R_k for test instance x from dictionary (list)
def R_k_inst(x):
    return RCPSP_inst_dict_named[sName[x]].iloc[1, 0:k_inst(x)].tolist()
# Function for extracting d_i for test instance x from dictionary (list)
def d_i_inst(x):
    return RCPSP_inst_dict_named[sName[x]].iloc[2:2+n_inst(x), 0].tolist()
# Function for extracting r_i_k for test instance x from dictionary (df)
def r_i_k_inst(x):
    return RCPSP_inst_dict_named[sName[x]].iloc[2:2+n_inst(x), 1:k_inst(x)+1]
# Function for extracting s_i for test instance x from dictionary (list)
def s_i_inst(x):
    return RCPSP_inst_dict_named[sName[x]].iloc[2:2+n_inst(x)-1, k_inst(x)+1].tolist()
# Function for extracting j^i_s for test instance x from dictionary (df)
def j_i_s_inst(x):
    return RCPSP_inst_dict_named[sName[x]].iloc[2:2+n_inst(x)-1, k_inst(x)+2:len(RCPSP_inst_dict_named[sName[x]].columns)]

def arcs(x):
    a = j_i_s_inst(x)
    a.index = range(1,len(a)+1)
    for r in len(a):
        for c in len(a.columns):
            b = [(r, a.i[])]
    return a


# Model Building for instance 1
# Model
model_SDT = Model("RCPSP: Time-Indexed Formulation with Step Variables and aggregated precedence constraints")

model_SDDT = Model("RCPSP: Time-Indexed Formulation with Step Variables and disaggregated precedence constraints")

# Using naive approach to create set t
ES_i = 0
LS_i = sum(d_i_inst(1))

# Variables
#for i in range(1, n_inst(1)):
#    for t in range(ES_i, LS_i):
y = model_SDDT.addVars(n_inst(1),LS_i, vtype = GRB.BINARY, name = "step variable")

# Objective Function
model_SDDT.setObjective(quicksum(t * (y[n_inst(1)][t]-y[n_inst(1)][t-1]) for t in range(ES_i+1, LS_i))+y[n_inst(1)][ES_i])

# Constraints
# model_SDDT.addConstrs((y[i, t] == 0 for i in range(0, n_inst(1)) for t in range(0, ES_i-1)), name="no starting before ES_i")
model_SDDT.addConstrs((y[i, LS_i] == 1 for i in range(0, n_inst(1))), name="all activities have started at LS_i")
model_SDDT.addConstrs(for (i,s) in arcs(1))


