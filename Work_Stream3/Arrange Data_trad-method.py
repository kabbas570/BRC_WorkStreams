
import pandas as pd

drug_arranged = pd.read_csv(r"C:\My_Data\BRC_Project\data\workstream_3\ASCOT Data Stata Files\Drug_Arranged.csv")
follow_2 = pd.io.stata.read_stata(r"C:\My_Data\BRC_Project\data\workstream_3\ASCOT Data Stata Files\DERI_followu2_working_reduced.dta")
data_baseline = pd.io.stata.read_stata(r"C:\My_Data\BRC_Project\data\workstream_3\ASCOT Data Stata Files\DERI_baseline_mw_endpan_mw_random.dta")

## aplly two filters ####
data_baseline = data_baseline[data_baseline['Antihyp'] == 'No'] ## filterthose with No prior medication  (3666)
data_baseline = data_baseline[data_baseline['anyevent'] == 'No'] ## filterthose with No fatal/non-fatal CV event (2698)


max_ = drug_arranged['TotDDos'].max()
min_ = drug_arranged['TotDDos'].min()

print(f"Maximum value in 'sbmean' column: {max_}")
print(f"Minimum value in 'sbmean' column: {min_}")



for column in drug_arranged.columns:
    column_type = drug_arranged[column].dtype
    print(column,'  ',column_type)
    if drug_arranged[column].isnull().any():
        if column_type == 'float32' or column_type == 'float64':
            mean_value = drug_arranged[column].mean()
            drug_arranged[column].fillna(mean_value, inplace=True)
        
        if column_type == 'category' or column_type == 'object':
            mode_value = drug_arranged[column].mode()[0]  # Get the mode (first value, as there might be multiple modes)
            drug_arranged[column].fillna(mode_value, inplace=True)
            
for column in follow_2.columns:
    column_type = follow_2[column].dtype
    print(column,'  ',column_type)
    if follow_2[column].isnull().any():
        if column_type == 'float32' or column_type == 'float64':
            mean_value = follow_2[column].mean()
            follow_2[column].fillna(mean_value, inplace=True)
        
        if column_type == 'category' or column_type == 'object':
            mode_value = follow_2[column].mode()[0]  # Get the mode (first value, as there might be multiple modes)
            follow_2[column].fillna(mode_value, inplace=True)
            
for column in data_baseline.columns:
    column_type = data_baseline[column].dtype
    print(column,'  ',column_type)
    if data_baseline[column].isnull().any():
        if column_type == 'float32' or column_type == 'float64':
            mean_value = data_baseline[column].mean()
            data_baseline[column].fillna(mean_value, inplace=True)
        
        if column_type == 'category' or column_type == 'object':
            mode_value = data_baseline[column].mode()[0]  # Get the mode (first value, as there might be multiple modes)
            data_baseline[column].fillna(mode_value, inplace=True)
            
            


unique_subjids = data_baseline['SubjID'].unique()
unique_subjids = unique_subjids[0:5]

map_1 = {'Amlodipine': 0, 'Perindopril': 1, 'Atenolol': 2, 'BFZ/K+': 3, 'Doxazosin': 4}
def map_med(Drugs, Dose):
    result = [0] * len(map_1)  # Initialize dosage list with zeros
    
    for drug, dose in zip(Drugs, Dose):
        if drug in map_1:
            index = map_1[drug]
            result[index] = dose
    return result

df_list = []


import os
folder_path = r'C:\My_Data\BRC_Project\data\workstream_3\data_csv\data1\F1\train'
csv_files = []
for filename in os.listdir(folder_path):
    if filename.endswith('.csv'):
        csv_files.append(filename[:-4])
#csv_files = csv_files[0:5]

for subj_id in csv_files:
    drug_records = drug_arranged[drug_arranged['SubjID'] == subj_id]
    follow_records = follow_2[follow_2['SubjID'] == subj_id]
    
    num_visit = len(follow_records)
    num_drug = len(drug_records)
    
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
    
    data = {
            "subj_ID": [subj_id],
            "sex": [sex],
            "race": [Race],
            "Vascular": [vascular],
            "Coronary": [Coronary],
            "Diabsube": [Diabsube]
        }
        
    
    for v in range(3+1):
        visit_col = f'visit_{v+1}'
       
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
                    
        drug_doze_OneHot = map_med(all_drug_at_thisVisit, Drug_Dose_thisVisit)
        drug_time_OneHot = map_med(all_drug_at_thisVisit, Drug_Days)

        
        column_name = f"age_{visit_col}"
        data[column_name] = [age] * len(data["subj_ID"])
        
        column_name = f"bmi_{visit_col}"
        data[column_name] = [bmi] * len(data["subj_ID"])
        
        for en in range(len(drug_doze_OneHot)):
            column_name = f"days_{en}_{visit_col}"
            data[column_name] = drug_doze_OneHot[en] * len(data["subj_ID"])
        
        for en in range(len(drug_time_OneHot)):            
            column_name = f"dose_{en}_{visit_col}"
            data[column_name] = drug_time_OneHot[en] * len(data["subj_ID"])
            
        sbpmean = SBP_Mean[v]
        
        column_name = f"sbpmean_{visit_col}"
        data[column_name] = [sbpmean] * len(data["subj_ID"])
            
    
    
    name = subj_id
    df = pd.DataFrame(data)
    df_list.append(df)
    #print(name)
    #file_path = r'C:\My_Data\BRC_Project\data\workstream_3\ASCOT Data Stata Files\per_patient/'+name +'.csv'

# Save the merged DataFrame to the CSV file
    #results_df3.to_csv(file_path, index=False)


merged_df = pd.concat(df_list, ignore_index=True)

merged_df = merged_df[['subj_ID','sex','race','Vascular','Coronary','Diabsube','age_visit_1','age_visit_2','age_visit_3','age_visit_4'
                       ,'bmi_visit_1','bmi_visit_2','bmi_visit_3','bmi_visit_4'
                       ,'days_0_visit_1','days_1_visit_1','days_2_visit_1','days_3_visit_1'
                       ,'days_0_visit_2','days_1_visit_2','days_2_visit_2','days_3_visit_2'
                       ,'days_0_visit_3','days_1_visit_3','days_2_visit_3','days_3_visit_3'
                       ,'days_0_visit_4','days_1_visit_4','days_2_visit_4','days_3_visit_4'
                       
                       ,'dose_0_visit_1','dose_1_visit_1','dose_2_visit_1','dose_3_visit_1'
                       ,'dose_0_visit_2','dose_1_visit_2','dose_2_visit_2','dose_3_visit_2'
                       ,'dose_0_visit_3','dose_1_visit_3','dose_2_visit_3','dose_3_visit_3'
                       ,'dose_0_visit_4','dose_1_visit_4','dose_2_visit_4','dose_3_visit_4'
                       
                       
                       ,'sbpmean_visit_1','sbpmean_visit_2','sbpmean_visit_3','sbpmean_visit_4']]


file_path = r'C:\My_Data\BRC_Project\data\workstream_3\data_csv\data1\F1/'+'train' +'.csv'


merged_df.to_csv(file_path, index=False)
