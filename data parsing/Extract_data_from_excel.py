
# coding: utf-8

# In[41]:

import pandas as pd
import os
os.getcwd()
os.chdir('C:/Users/yulong/Desktop/BUSN capstone/PECO data-20171029T204031Z-001/PECO data/')
writer = pd.ExcelWriter('49_oct_2015_may_2016.xlsx')
for dirpath, dirnames, filenames in os.walk('49 Series Reports_ Oct 2015 to May 2016'):
    for x in range(len(filenames)):
        
        df = pd.read_excel(dirpath+'/'+filenames[x],parse_dates = False)

        df = df[1:]
        new_list_witn_na = []
        for i,j in enumerate(df.count(axis=1)):
            if j==2 or j==1 or j==3:

                new_list_witn_na.append(df.loc[i+1].values.tolist())
        new_list = []
        for j in range(len(new_list_witn_na)):
            ll = [i for i in new_list_witn_na[j] if not(i != i)]
            new_list.append(ll)
        new_list = list(filter(None, new_list))
        list3=[]
        dict1 = {}
        for i in new_list: 
            if '-' in i[0] and len(i[0])==6:
                if len(dict1) >1 and 'Summary' not in dict1.keys():
                    list3.append(dict1)
                series = i[0]
                dict1={}
            if len(i) ==2:                
                dict1 ['series']=series
                dict1[i[len(i)-2]] = i[len(i)-1]
            if len(i) ==3:
                dict1['Summary']=i[0]
                dict1 ['series']=series
                dict1[i[len(i)-2]] = i[len(i)-1]

        df_final = pd.DataFrame(list3)  
        
        sheet_name = filenames[x]
        sheet_name = sheet_name.replace('.','_')
        sheet_name = sheet_name.split('_')[5:11]
        sheet_name = '-'.join(sheet_name)
        df_final.to_excel(writer,sheet_name)
writer.save()


# In[ ]:



