import pandas as pd

drug_arranged = pd.read_csv("/data/scratch/acw676/BRC/WS_3/org_data/Drug_Arranged.csv")
follow_2 = pd.io.stata.read_stata("/data/scratch/acw676/BRC/WS_3/org_data/DERI_followu2_working_reduced.dta")
data_baseline = pd.io.stata.read_stata("/data/scratch/acw676/BRC/WS_3/org_data/DERI_baseline_mw_endpan_mw_random.dta")

## aplly two filters ####
data_baseline = data_baseline[data_baseline['Antihyp'] == 'No'] ## filterthose with No prior medication  (3666)
data_baseline = data_baseline[data_baseline['anyevent'] == 'No'] ## filterthose with No fatal/non-fatal CV event (2698)

drug_ata = data_baseline[data_baseline['TRTCDA'] == 'Atenolol'] ## filterthose with No fatal/non-fatal CV event (2698)
drug_aml = data_baseline[data_baseline['TRTCDA'] == 'Amlodipine'] ## filterthose with No fatal/non-fatal CV event (2698)
data_baseline = drug_aml

            
for column in follow_2.columns:
    column_type = follow_2[column].dtype
    #print(column,'  ',column_type)
    if follow_2[column].isnull().any():
        if column_type == 'float32' or column_type == 'float64':
            mean_value = follow_2[column].mean()
            follow_2[column].fillna(mean_value, inplace=True)
        
        if column_type == 'category' or column_type == 'object':
            mode_value = follow_2[column].mode()[0]  # Get the mode (first value, as there might be multiple modes)
            follow_2[column].fillna(mode_value, inplace=True)
            
for column in data_baseline.columns:
    column_type = data_baseline[column].dtype
    #print(column,'  ',column_type)
    if data_baseline[column].isnull().any():
        if column_type == 'float32' or column_type == 'float64':
            mean_value = data_baseline[column].mean()
            data_baseline[column].fillna(mean_value, inplace=True)
        
        if column_type == 'category' or column_type == 'object':
            mode_value = data_baseline[column].mode()[0]  # Get the mode (first value, as there might be multiple modes)
            data_baseline[column].fillna(mode_value, inplace=True)
            
            


unique_subjids = data_baseline['SubjID'].unique()
#unique_subjids = unique_subjids[0:5]

map_1 = {'Amlodipine': 0, 'Perindopril': 1, 'Atenolol': 2, 'BFZ/K+': 3, 'Doxazosin': 4}
def map_med(Drugs, Dose):
    result = [0] * len(map_1)  # Initialize dosage list with zeros
    
    for drug, dose in zip(Drugs, Dose):
        if drug in map_1:
            index = map_1[drug]
            result[index] = dose
    return result

df_list = []

import math
def contains_nan(nested_list):
    for sublist in nested_list:
        for element in sublist:
            try:
                if math.isnan(element):
                    return True
            except TypeError:
                continue
    return False

for subj_id in unique_subjids:
    
    chk = []
    
    drug_records = drug_arranged[drug_arranged['SubjID'] == subj_id]
    follow_records = follow_2[follow_2['SubjID'] == subj_id]
    
    num_visit = len(follow_records)
    num_drug = len(drug_records)
    
    results_df3 = pd.DataFrame(columns=['SubjID'])
    
    
    ### for static data append ###
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    
    ### for Dynamic Vailabes ##
    
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True) # age
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True) # bmi

    ### for drug OneHot append 
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    
    
    ### for days_diff OneHot append 
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    
    ### for Targert BP
    results_df3 = results_df3.append({'SubjID': subj_id}, ignore_index=True)
    
    first_sbpmean = data_baseline[data_baseline['SubjID'] == subj_id].iloc[0]['sbpmean']
    follow_sbpmean = follow_records['sbpmean']
    
    SBP_Mean = pd.Series([first_sbpmean]).append(follow_sbpmean, ignore_index=True)
            
    
    ## static variables ###
    
    sex = data_baseline[data_baseline['SubjID'] == subj_id].iloc[0]['Sex']
    Race = data_baseline[data_baseline['SubjID'] == subj_id].iloc[0]['RaceC']
    vascular = data_baseline[data_baseline['SubjID'] == subj_id].iloc[0]['Vascular']
    Coronary = data_baseline[data_baseline['SubjID'] == subj_id].iloc[0]['Coronary']
    Diabsube = data_baseline[data_baseline['SubjID'] == subj_id].iloc[0]['DIABSUBE']


    #dob = data_baseline.loc[subj_id,'dob']
    dob = pd.to_datetime(data_baseline[data_baseline['SubjID']== subj_id].iloc[0]['dob'], errors='coerce')
    height_baseline = data_baseline[data_baseline['SubjID'] == subj_id].iloc[0]['height_']/100
    weight_baseline = data_baseline[data_baseline['SubjID'] == subj_id].iloc[0]['weight_']
    folow_weight = follow_records['weight_']
    
    vis_weight = pd.Series([weight_baseline]).append(folow_weight, ignore_index=True)

    first_vis = data_baseline[data_baseline['SubjID'] == subj_id].iloc[0]['vis_date']
    
    folow_vis = follow_records['vis_date']
    vis_dates = pd.Series([first_vis]).append(folow_vis, ignore_index=True)
    
    
    for v in range(num_visit+1):
        visit_col = f'visit_{v+1}'
        results_df3.loc[0, visit_col] = [sex][0]
        results_df3.loc[1, visit_col] = [Race][0]
        results_df3.loc[2, visit_col] = [vascular][0]
        results_df3.loc[3, visit_col] = [Coronary][0]
        results_df3.loc[4, visit_col] = [Diabsube][0]
        
        all_drug_at_thisVisit = []
        Drug_Dose_thisVisit = []
        Drug_Days = []
        
        globals()['viz_date_' + str(v)] = pd.to_datetime(vis_dates.iloc[v], errors='coerce')
        age = (globals()['viz_date_' + str(v)]  - dob)
        age  = round((age.days)/365.25, 2)
        
        bmi = vis_weight[v]/(height_baseline**2)
        bmi  = round(bmi, 2)

        
        for d in range(num_drug):
            first_drug_record = drug_records.iloc[d]        
            start_an = pd.to_datetime(first_drug_record['startAn'], errors='coerce')
            stop_an = pd.to_datetime(first_drug_record['stopAn'], errors='coerce')
            globals()['start_an_' + str(d)] = pd.to_datetime(first_drug_record['startAn'], errors='coerce')
            globals()['stop_an_' + str(d)] = pd.to_datetime(first_drug_record['stopAn'], errors='coerce')
            globals()['drug_stc_' + str(d)]  = drug_records.iloc[d]['drugStC']
            globals()['TotDDos_' + str(d)]  = drug_records.iloc[d]['TotDDos']
 
            if globals()['start_an_' + str(d)] < globals()['viz_date_' + str(v)]: 
                
                if globals()['stop_an_' + str(d)] >= globals()['viz_date_' + str(v)]: 
                                
                    drug_1 = globals()['drug_stc_' + str(d)]
                    dose_1 =  globals()['TotDDos_' + str(d)]
                    #print(subj_id,' ->',globals()['drug_stc_' + str(d)],globals()['TotDDos_' + str(d)] )
                    all_drug_at_thisVisit.append(drug_1)
                    Drug_Dose_thisVisit.append(dose_1)
                    
                    days_1 = globals()['viz_date_' + str(v)] - globals()['start_an_' + str(d)]
                    days = days_1.days
                    Drug_Days.append(days)
        
        #print(all_drug_at_thisVisit)
        chk.append(all_drug_at_thisVisit)
        drug_doze_OneHot = map_med(all_drug_at_thisVisit, Drug_Dose_thisVisit)
        drug_time_OneHot = map_med(all_drug_at_thisVisit, Drug_Days)
        results_df3.loc[5, visit_col] = age
        results_df3.loc[6, visit_col] = bmi
        for en in range(len(drug_doze_OneHot)):
            results_df3.loc[en + 7, visit_col] = drug_doze_OneHot[en]
        
        for en in range(len(drug_time_OneHot)):
            results_df3.loc[en + 7+5, visit_col] = drug_time_OneHot[en]
                
        sbpmean = SBP_Mean[v]
        results_df3.loc[17, visit_col] = sbpmean
        
                
    name = subj_id
    cond = contains_nan(chk)    
    if not cond:      
      file_path = '/data/scratch/acw676/BRC/C_Data/aml/'+name +'.csv'
      results_df3.to_csv(file_path, index=False)
