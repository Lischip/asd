import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set_theme(style="whitegrid")
from enum import Enum


def _processing(df, scenarios):
    global namedict
    df = df.loc[df.index.str.split("_").str[-1].isin(scenarios)].T
    df.columns = df.columns.to_series().apply(lambda value: namedict[value.split("_")[-1]])
    df.index = df.index.astype("float")
    return df


def _coi_processing(df, scenarios):
    df = _processing(df, scenarios)
    # get each year
    df = df[df.index % 1 == 0].copy()
    # only care about the difference
    return df.diff(axis=0)


def _meat_processing(df, scenarios):
    df = _processing(df, scenarios)
    if df.empty:
        return df
    min_idx, max_idx = min(df.index), max(df.index)
    # for truncation
    df = df.groupby(np.arange(len(df)) // 4, as_index=False).mean()
    idx = pd.DataFrame(np.arange(min_idx, max_idx, (max_idx - min_idx) / df.shape[0]))
    df = pd.concat([idx, df], axis=1)
    df.set_index(0, inplace=True)
    df.index.name = None
    df.index = df.index.astype("float")
    return df


def data_processing(loc):
    location = "../data/" + loc + ".txt"
    df = pd.read_csv(location, delimiter="\t", dtype="string")
    df = df.iloc[:, :-1]
    # Ok, so as we all know Python is retarded when it comes to memory allocation, which means we have to resort ugly constructs like this
    # C++ for the win
    temp = df.iloc[:, 1:].applymap(lambda value: float(value.replace("M", '')) * 1000000 if "M" in value else value)
    temp = temp.apply(pd.to_numeric)
    df = pd.concat([df.iloc[:, 0], temp], axis=1)
    # df = pd.DataFrame(df.iloc[:,0]).join(temp)
    df.set_index("Date", inplace=True)
    return df


def get_data(data, df, scenarios):
    if data.name in [DATA.COI.name, DATA.DALY.name]:
        df = pd.DataFrame(df.loc[data_df.index.str.contains(data.value)])
        return _coi_processing(df, scenarios)
    elif data.name in [DATA.COIACC.name, DATA.DALYACC.name]:
        df = pd.DataFrame(df.loc[data_df.index.str.contains(data.value[1:])])
        return _meat_processing(df, scenarios)
    else:
        df = pd.DataFrame(df.loc[data_df.index.str.contains(data.value)])
        return _meat_processing(df, scenarios)

#What's the name of ur file?
loc = "cb_data"

policydict = {"cb" : "Consumption behaviour",
             "fs" : "Food safety and handling",
             "pc": "Fly population control",
             "ec": "Exposure control",
             "ss" : "Safe slaughtering",
             "00": "No policies"}

#What's ur policy called?
policy = policydict[loc[:2]]

bw = ["base", "12"]
po = ["1", "2", "3"]
t = ["4", "5", "6"]
s = ["7","8", "9"]
pu = ["10", "11"]

scenario_dict= {"Base and Worst Case": bw,
                "Population": po,
                "Temperature": t,
                "Seasonal": s,
                "Public health": pu}

# What scenarios do u want plotted from policy?
scenarioss = [bw, po, t,s, pu]

namedict = {"base": "Base run",
            "1": "Population: medium increase",
            "2": "Population: large increase",
            "3": "Population: decrease",
            "4": "Temperature: increase 1 degree",
            "5": "Temperature: increase 1.5 degrees",
            "6": "Temperature: increase 2 degrees",
            "7": "Seasonality: no temperature change",
            "8": "Seasonality: Fast temperature change",
            "9": "Seasonality: linear temperature change",
            "10": "Public health: 10% fewer symptoms",
            "11": "Public health: 10% more symptoms",
            "12": "Worst case"}

class DATA(Enum):
    COI = "Cost of Illness"
    DALY = "DALY"
    COIACC = "aCost of Illness"
    DALYACC = "aDALY"
    MEAT = "contaminated meat"
    ENVH = "human infection from environment"
    ENVC = "rate of chicken infection from environment"

# Because we like colour consistency
simulcmap = "tab10"
cmap = plt.cm.get_cmap(simulcmap, 10)
cmapcolors = cmap(range(10))

colordict = {namedict["base"] : cmapcolors[0],
            namedict["base"] + ", w/ policy" : cmapcolors[0],
            namedict["base"] + ", w/o policy": cmapcolors[0],
            namedict["1"] : cmapcolors[0],
            namedict["1"] + ", w/ policy"  : cmapcolors[0],
            namedict["1"] + ", w/o policy" : cmapcolors[0],
            namedict["2"] :  cmapcolors[1],
            namedict["2"] + ", w/ policy"  :  cmapcolors[1],
            namedict["2"] + ", w/o policy" : cmapcolors[1],
            namedict["3"] :  cmapcolors[2],
            namedict["3"] + ", w/ policy" : cmapcolors[2],
            namedict["3"] + ", w/o policy" : cmapcolors[2],
            namedict["4"] :  cmapcolors[3],
            namedict["4"] + ", w/ policy" :  cmapcolors[3],
            namedict["4"]  + ", w/o policy"  :  cmapcolors[3],
            namedict["5"] :  cmapcolors[0],
            namedict["5"] + ", w/ policy" : cmapcolors[0],
            namedict["5"]  + ", w/o policy"  : cmapcolors[0],
            namedict["6"] :  cmapcolors[4],
            namedict["6"] + ", w/ policy" :  cmapcolors[4],
            namedict["6"] + ", w/o policy" :  cmapcolors[4],
            namedict["7"] :  cmapcolors[5],
            namedict["7"] + ", w/ policy" : cmapcolors[5],
            namedict["7"] + ", w/o policy"  : cmapcolors[5],
            namedict["8"] : cmapcolors[6],
            namedict["8"] + ", w/ policy" :  cmapcolors[6],
            namedict["8"] + ", w/o policy"  :  cmapcolors[6],
            namedict["9"] :  cmapcolors[0],
            namedict["9"] + ", w/ policy" : cmapcolors[0],
            namedict["9"] + ", w/o policy"  : cmapcolors[0],
            namedict["10"] :  cmapcolors[7],
            namedict["10"] + ", w/ policy" : cmapcolors[7],
            namedict["10"] + ", w/o policy"  : cmapcolors[7],
            namedict["11"] :  cmapcolors[9],
            namedict["11"] + ", w/ policy" : cmapcolors[9],
            namedict["11"] + ", w/o policy"  : cmapcolors[9],
            namedict["12"] : cmapcolors[8],
            namedict["12"] + ", w/ policy" : cmapcolors[8],
            namedict["12"] + ", w/o policy"  :  cmapcolors[8]}

for loc in ("00_data", "cb_data", "fs_data", "pc_data", "ec_data", "ss_data"):
    policy = policydict[loc[:2]]
    for name, scenarios in scenario_dict.items():
        base_scenarios = scenarios
        num_scen = len(scenarios) if len(scenarios) > len(base_scenarios) else len(base_scenarios)

        data_df = data_processing(loc)

        base_df = data_processing("00_data")


        #cost of illness
        policy_coi = get_data(DATA.COI, data_df, scenarios)
        base_coi = get_data(DATA.COI, base_df, base_scenarios)
        policy_coi = policy_coi.reindex(base_coi.columns, axis=1)

        for df in [policy_coi, base_coi]:
            for column in policy_coi.columns:
                df[column] /= 1000000

        policy_coi = policy_coi.add_suffix(', w/ policy')
        base_coi = base_coi.add_suffix(', w/o policy')

        fig, ax = plt.subplots(figsize=(10, 10))

        if policy_coi.shape[1] >= 1:
            sns.lineplot(data=policy_coi[2022:], palette=colordict, dashes=[(1, 0)] * len(scenarios))
            #sns.lineplot(data=policy_coi[2022:], dashes=[(1, 0)] * len(scenarios))
        if base_coi.shape[1] >= 1:
            sns.lineplot(data=base_coi[2022:], palette=colordict, dashes=[(4, 2)] * len(base_scenarios))
            #sns.lineplot(data=base_coi[2022:], dashes=[(4, 2)] * len(base_scenarios))

        plt.xlabel('Year');
        plt.ylabel('Cost (million Euro)')
        plt.title(policy + ': Cost of Illness')
        plt.legend(title="Scenarios", ncol=2, fancybox=True, bbox_to_anchor=(0, -0.13 - (0.02 * num_scen), 1, 1),
                   loc="lower center")

        plt.savefig("../images/" + loc[:2] + "_" + name +"_coi.png", dpi=300, bbox_inches='tight')
        plt.show()
        del policy_coi
        del base_coi

        # contaminated chicken meat

        policy_meat = get_data(DATA.MEAT, data_df, scenarios)
        base_meat = get_data(DATA.MEAT, base_df, base_scenarios)
        policy_meat = policy_meat.reindex(base_meat.columns, axis=1)

        for df in [policy_meat, base_meat]:
            for column in policy_meat.columns:
                df[column] /= 1000000

        policy_meat = policy_meat.add_suffix(', w/ policy')
        base_meat = base_meat.add_suffix(', w/o policy')

        fig, ax = plt.subplots(figsize=(10, 10))

        if policy_meat.shape[1] >= 1:
            sns.lineplot(data=policy_meat.loc[policy_meat.index >= 2021.75], palette=colordict, dashes=[(1, 0)] * len(scenarios))
            #sns.lineplot(data=policy_meat.loc[policy_meat.index >= 2021.75], dashes=[(1, 0)] * len(scenarios))
        if base_meat.shape[1] >= 1:
            sns.lineplot(data=base_meat.loc[base_meat.index >= 2021.75], palette=colordict, dashes=[(4, 2)] * len(base_scenarios))
            #sns.lineplot(data=base_meat.loc[base_meat.index >= 2021.75], dashes=[(4, 2)] * len(base_scenarios))

        plt.xlabel('Year');
        plt.ylabel('Chicken meat (million kg)')
        plt.title(policy + ': Contaminated chicken meat')
        plt.legend(title="Scenarios", ncol=2, fancybox=True, bbox_to_anchor=(0, -0.13 - (0.02 * num_scen), 1, 1),
                   loc="lower center")
        ax.grid(True)
        plt.savefig("../images/" + loc[:2] + "_" + name +"_meat.png", dpi=300, bbox_inches='tight')
        plt.show()
        del policy_meat
        del base_meat

        #accumulated cost of illness

        policy_coi = get_data(DATA.COIACC, data_df, scenarios)
        base_coi = get_data(DATA.COIACC, base_df, base_scenarios)
        policy_coi = policy_coi.reindex(base_coi.columns, axis=1)

        for df in [policy_coi, base_coi]:
            for column in policy_coi.columns:
                df[column] /= 1000000

        policy_coi = policy_coi.add_suffix(', w/ policy')
        base_coi = base_coi.add_suffix(', w/o policy')

        fig, ax = plt.subplots(figsize=(10, 10))

        if policy_coi.shape[1] >= 1:
            sns.lineplot(data=policy_coi.loc[policy_coi.index >= 2021.75], dashes=[(1, 0)] * len(scenarios), palette=colordict)
        if base_coi.shape[1] >= 1:
            sns.lineplot(data=base_coi.loc[base_coi.index >= 2021.75], dashes=[(4, 2)] * len(base_scenarios), palette=colordict)

        plt.xlabel('Year');
        plt.ylabel('Cost (million euro)')
        plt.title(policy + ': Accumulated Cost of Illness')
        plt.legend(title="Scenarios", ncol=2, fancybox=True, bbox_to_anchor=(0, -0.16 - (0.02 * num_scen), 1, 1),
                   loc="lower center")
        ax.grid(True)
        plt.savefig("../images/" + loc[:2] + "_" + name  + "_acoi.png", dpi=300, bbox_inches='tight')
        plt.show()
        del policy_coi
        del base_coi

        #human infection

        policy_envh = get_data(DATA.ENVH, data_df, scenarios)
        base_envh = get_data(DATA.ENVH, base_df, base_scenarios)
        policy_envh = policy_envh.reindex(base_envh.columns, axis=1)
        policy_envh = policy_envh.add_suffix(', w/ policy')
        base_envh = base_envh.add_suffix(', w/o policy')

        fig, ax = plt.subplots(figsize=(10, 10))

        sns.lineplot(data=policy_envh.loc[policy_envh.index >= 2021.75], dashes=[(1, 0)] * len(scenarios), palette=colordict)
        sns.lineplot(data=base_envh.loc[base_envh.index >= 2021.75], dashes=[(4, 2)] * len(base_scenarios), palette=colordict)

        plt.xlabel('Year');
        plt.ylabel('Persons')
        plt.title(policy + ': Humans infected by environment')
        plt.legend(title="Scenarios", ncol=2, fancybox=True, bbox_to_anchor=(0, -0.16 - (0.02 * num_scen), 1, 1),
                   loc="lower center")
        ax.grid(True)
        plt.savefig("../images/" + loc[:2] + "_" + name + "_humaninfection.png", dpi=300, bbox_inches='tight')
        plt.show()
        del policy_envh
        del base_envh

        # chicken infection
        policy_envc = get_data(DATA.ENVC, data_df, scenarios)
        base_envc = get_data(DATA.ENVC, base_df, base_scenarios)
        policy_envc = policy_envc.reindex(base_envc.columns, axis=1)
        policy_envc = policy_envc.add_suffix(', w/ policy')
        base_envc = base_envc.add_suffix(', w/o policy')

        fig, ax = plt.subplots(figsize=(10, 10))

        sns.lineplot(data=policy_envc.loc[policy_envc.index >= 2021.75], dashes=[(1, 0)] * len(scenarios), palette=colordict)
        sns.lineplot(data=base_envc.loc[base_envc.index >= 2021.75], dashes=[(4, 2)] * len(base_scenarios), palette=colordict)

        plt.xlabel('Year');
        plt.ylabel('Ratio')
        plt.title(policy + ': Ratio of chickens infected by environment')
        plt.legend(title="Scenarios", ncol=2, fancybox=True, bbox_to_anchor=(0, -0.16 - (0.02 * num_scen), 1, 1),
                   loc="lower center")
        ax.grid(True)
        plt.savefig("../images/" + loc[:2] + "_" + name + "_chickeninfection.png", dpi=300, bbox_inches='tight')
        plt.show()

        del policy_envc
        del base_envc

        #daly
        policy_daly = get_data(DATA.DALY, data_df, scenarios)
        base_daly = get_data(DATA.DALY, base_df, base_scenarios)
        policy_daly = policy_daly.reindex(base_daly.columns, axis=1)
        policy_daly = policy_daly.add_suffix(', w/ policy')
        base_daly = base_daly.add_suffix(', w/o policy')

        fig, ax = plt.subplots(figsize=(10, 10))

        if policy_daly.shape[1] >= 1:
            sns.lineplot(data=policy_daly[2022:], dashes=[(1, 0)] * len(scenarios), palette=colordict)
        if base_daly.shape[1] >= 1:
            sns.lineplot(data=base_daly[2022:], dashes=[(4, 2)] * len(base_scenarios), palette=colordict)

        plt.xlabel('Year');
        plt.ylabel('Euro')
        plt.title(policy + ': DALYs')

        plt.legend(title="Scenarios", ncol=2, fancybox=True, bbox_to_anchor=(0, -0.16 - (0.02 * num_scen), 1, 1),
                   loc="lower center")
        ax.grid(True)

        plt.savefig("../images/" + loc[:2] + "_" + name + "_daly.png", dpi=300, bbox_inches='tight')
        plt.show()

        #accumulated daly

        policy_daly = get_data(DATA.DALYACC, data_df, scenarios)
        base_daly = get_data(DATA.DALYACC, base_df, base_scenarios)
        policy_daly = policy_daly.reindex(base_daly.columns, axis=1)
        policy_daly = policy_daly.add_suffix(', w/ policy')
        base_daly = base_daly.add_suffix(', w/o policy')

        fig, ax = plt.subplots(figsize = (10,10))

        if policy_daly.shape[1] >= 1:
            sns.lineplot(data=policy_daly.loc[policy_daly.index >= 2021.75],  dashes=[(1, 0)] * len(scenarios), palette=colordict)
        if base_daly.shape[1] >= 1:
            sns.lineplot(data=base_daly.loc[base_daly.index >= 2021.75],  dashes=[(4, 2)] * len(base_scenarios), palette=colordict)

        plt.xlabel('Year'); plt.ylabel('Euro')
        plt.title(policy + ': Accumulated DALYs')
        plt.legend(title="Scenarios", ncol=2, fancybox=True, bbox_to_anchor=(0, -0.16-(0.02*num_scen), 1, 1), loc="lower center")
        ax.grid(True)
        plt.savefig("../images/" + loc[:2] + "_" + name + "_adaly.png", dpi=300, bbox_inches='tight')
        plt.show()
        del policy_daly
        del base_daly

print("Done!")