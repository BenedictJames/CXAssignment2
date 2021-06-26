import pandas as pd
from gurobipy import *
import math

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
    return df.rename(columns=dict)

# Apply function to dictionary containing all RCPSP test instances as dfs; this will name the dfs' columns
RCPSP_inst_dict_named = {k: rename_columns(v, names_dict) for k, v in RCPSP_inst_dict.items()}


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
    return RCPSP_inst_dict_named[sName[x]].iloc[2:2 + n_inst(x), 0].tolist()


# Function for extracting r_i_k for test instance x from dictionary (df)
def r_i_k_inst(x):
    return RCPSP_inst_dict_named[sName[x]].iloc[2:2 + n_inst(x), 1:k_inst(x) + 1]


# Function for extracting s_i for test instance x from dictionary (list)
def s_i_inst(x):
    return RCPSP_inst_dict_named[sName[x]].iloc[2:2 + n_inst(x) - 1, k_inst(x) + 1].tolist()


# Function for extracting j^i_s for test instance x from dictionary (df)
def j_i_j_inst(x):
    return RCPSP_inst_dict_named[sName[x]].iloc[2:2 + n_inst(x) - 1,
           k_inst(x) + 2:len(RCPSP_inst_dict_named[sName[x]].columns)]


# Create list with all precedence relations as tuples
def arcs(x):
    a = j_i_j_inst(x)
    a.index = range(1, len(a) + 1)
    b = []
    for r in range(0, len(a)):
        for c in range(0, len(a.columns)):
            if (math.isnan(a.iat[r, c]) == False): b.append((r, a.iat[r, c] - 1))
    return b

# Function to handle negative indices dv
def negative_index(dv, index_1, index_2):
    if index_2 >= 0:
        return dv[index_1, index_2]
    else:
        return 0

# Solving The Generic Time Scheduling Program
# Storing ES for each activity in each instance
df_ES = pd.DataFrame(0, index = range(n_inst(1)), columns = sName[1:])
# Model

for instance in range(1, len(RCPSP_inst_dict_named)+1):
    Generic = Model("Generic Time Scheduling Problem")
    Generic.setParam('TimeLimit', 20)
    Generic.setParam('LogtoConsole', 0)
    s = Generic.addVars(n_inst(instance), name="start times")

    # objective
    Generic.setObjective(quicksum(s[i] for i in range(n_inst(instance))), GRB.MINIMIZE)

    # constraints
    Generic.addConstrs((s[j] - s[i] >= d_i_inst(instance)[i] for (i, j) in arcs(instance)), "timelags")
    Generic.addConstr(s[0] == 0)
    Generic.addConstr(s[n_inst(instance) - 1] <= sum(d_i_inst(instance)))

    # solve the model
    Generic.optimize()
    # print("Number of jobs: " + str(n_inst(instance)))
    # print("Objective value: " + str(Generic.objVal))
    # Write model solutions to df with all ES
    a = Generic.getVars()
    for v in range(n_inst(instance)):
        df_ES.iloc[v, instance-1] = a[v].x


## RCPSP Models

# Models
model_SDT = Model("RCPSP: Time-Indexed Formulation with Step Variables and aggregated precedence constraints")
model_SDT.setParam('TimeLimit', 10*60)
model_SDT.setParam('LogtoConsole', 0)

model_SDDT = Model("RCPSP: Time-Indexed Formulation with Step Variables and disaggregated precedence constraints")
model_SDDT.setParam('TimeLimit', 10*60)
model_SDDT.setParam('LogtoConsole', 0)

#for instance in range(1, len(RCPSP_inst_dict_named)+1):
for instance in range(1, 2):

    ## Lower and Upper Bound for t
    # Using naive approach to create set t
    # ES_i = 0
    LS_i = sum(d_i_inst(instance))+1

    # Variables
    # for i in range(1, n_inst(1)):
    # for t in range(ES_i, LS_i):
    y = model_SDDT.addVars(n_inst(instance), LS_i, vtype=GRB.BINARY, name="step variable")

    # Objective Function
    model_SDDT.setObjective(quicksum(t * (y[n_inst(instance) - 1, t] - negative_index(y, n_inst(instance) - 1, t - 1)) for t in range(df_ES.iloc[n_inst(instance)-1, instance-1], LS_i)),
                           GRB.MINIMIZE)

    # Constraints

    # First Attempt to handle negative indices
    # For Constraints where DV y is indexed at t-d_i, it first has to checked that t-d_i does not turn negative
    # If t-d_i does not turn negative, the respective DV y is set to 0
    #for (i, j) in arcs(1):
    #    for t in range(ES_i, LS_i):
    #        if t - d_i_inst(1)[i] >= 0:
    #            model_SDDT.addConstr((y[i, t - d_i_inst(1)[i]] - y[j, t] >= 0),
    #                                 name="(2.10) disaggregated precedence constraint with t-d_i >=0")
    #        else:
    #            model_SDDT.addConstr((y[j, t] == 0), name="(2.10) disaggregated precedence constraint with t-d_i < 0")
    model_SDDT.addConstrs((negative_index(y, i, t - d_i_inst(instance)[i]) - y[j, t] >= 0
                           for (i, j) in arcs(instance)
                           for t in range(0, LS_i)),
                          name="(2.10) disaggregated precedence constraint")

    model_SDDT.addConstrs((quicksum(
        r_i_k_inst(instance).iloc[i, k] * (y[i, t] - negative_index(y, i, t - d_i_inst(instance)[i])) for i in range(n_inst(instance))) <=
                           R_k_inst(instance)[k]
                           for t in range(0, LS_i-1)
                           for k in range(k_inst(instance))),
                          name="(2.11) resource constraint")
    # Old Constraint failed due to index t-d_i_inst(1)[i] < 0
    # model_SDDT.addConstrs((quicksum(r_i_k_inst(1).iloc[i, k]*(y[i, t]-y[i, t-d_i_inst(1)[i]]) for i in range(0, n_inst(1))) <= R_k_inst(1)[k]
    #                       for t in range(ES_i, LS_i)
    #                       for k in range(0, k_inst(1))),
    #                      name="(2.11) ressource constraint")

    model_SDDT.addConstrs((y[i, LS_i - 1] == 1
                           for i in range(n_inst(instance))),
                          name="(2.12) all activities have started at LS_i")

    model_SDDT.addConstrs((y[i, t] - negative_index(y, i, t - 1) >= 0
                           for i in range(n_inst(instance))
                           for t in range(df_ES.iloc[i, instance-1], LS_i)),
                          name="(2.13) step variable cannot switch back to 0")

    model_SDDT.addConstrs((y[i, t] == 0
                           for i in range(n_inst(instance))
                           for t in range(df_ES.iloc[i, instance-1] - 1)),
                          name="(2.14) no starting before ES_i")

    model_SDDT.optimize()
    print("Objective value SDDT of test instance " + str(sName[instance]) + ": " + str(model_SDDT.objVal))
    print("Runtime model SDDT for test instance " + str(sName[instance]) + " in seconds: " + str(model_SDDT.runtime) + "s")
#    print("Gap to optimum for model SDDT for test instance " + str(sName[instance]) + " in seconds: " + str(model_SDDT.MIPGap) + "s")

    ## MODEL SDT

    y = model_SDT.addVars(n_inst(instance), LS_i, vtype=GRB.BINARY, name="step variable")

    # Objective Function
    model_SDT.setObjective(quicksum(t * (y[n_inst(instance) - 1, t] - negative_index(y, n_inst(instance) - 1, t - 1)) for t in range(df_ES.iloc[n_inst(instance)-1, instance-1], LS_i)),
                           GRB.MINIMIZE)


    # Constraints
    model_SDT.addConstrs((quicksum(t * (y[j, t] - negative_index(y, j, t - 1)) for t in range(0, LS_i)) -
                        quicksum(t * (y[i, t] - negative_index(y, i, t - 1)) for t in range(0, LS_i)) >= d_i_inst(instance)[i] for (i, j) in arcs(instance)),
                        name="(2.10) aggregated precedence constraint")

    model_SDT.addConstrs((quicksum(
        r_i_k_inst(instance).iloc[i, k] * (y[i, t] - negative_index(y, i, t - d_i_inst(instance)[i])) for i in range(n_inst(instance))) <=
                           R_k_inst(instance)[k]
                           for t in range(0, LS_i-1)
                           for k in range(k_inst(instance))),
                          name="(2.11) resource constraint")

    model_SDT.addConstrs((y[i, LS_i - 1] == 1
                           for i in range(n_inst(instance))),
                          name="(2.12) all activities have started at LS_i")

    model_SDT.addConstrs((y[i, t] - negative_index(y, i, t - 1) >= 0
                           for i in range(n_inst(instance))
                           for t in range(df_ES.iloc[i, instance-1], LS_i)),
                          name="(2.13) step variable cannot switch back to 0")

    model_SDT.addConstrs((y[i, t] == 0
                           for i in range(n_inst(instance))
                           for t in range(df_ES.iloc[i, instance-1] - 1)),
                          name="(2.14) no starting before ES_i")

    model_SDT.optimize()
    print("Objective value SDT of test instance " + str(sName[instance]) + ": " + str(model_SDT.objVal))
    print("Runtime model SDT for test instance " + str(sName[instance]) + " in seconds: " + str(model_SDT.runtime) + "s")
#    print("Gap to optimum for model SDT for test instance " + str(sName[instance]) + " in seconds: " + str(model_SDT.MIPGap) + "s")



