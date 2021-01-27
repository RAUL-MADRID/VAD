import pandas as pd
import numpy as np
from itertools import islice

# 데이터 정규화
def normalization(df) :
    x_df = df[['x1', 'x2', 'x3']]
    y_df = df[['y1', 'y2', 'y3']]
    normalized_x = (x_df - x_df.mean()) / x_df.std()
    normalized_df = pd.concat([normalized_x, y_df], axis = 1)
    
    return normalized_df


# 각 시나리오 file 별로 x1, x2, x3와 y1,y2,y3를 행단위로 이어준다. 
# 데이터 형태 :
# --------------
# | [x1]  [y1] |
# | [x2]  [y2] |
# | [x3]  [y3] |
# --------------
def infile_concat(df) : 
    person_1_x = pd.concat([pd.Series(np.ones(pd.Series(df['x1']).shape[0])), pd.Series(df['x1'])], axis = 1)
    person_2_x = pd.concat([2*pd.Series(np.ones(pd.Series(df['x2']).shape[0])), pd.Series(df['x2'])], axis = 1)
    person_3_x = pd.concat([3*pd.Series(np.ones(pd.Series(df['x3']).shape[0])), pd.Series(df['x3'])], axis = 1)
    
    person_1_y = pd.Series(df['y1'])
    person_2_y = pd.Series(df['y2'])
    person_3_y = pd.Series(df['y3'])
    
    total = pd.concat([person_1_x, person_2_x, person_3_x], axis = 1)
    
    a = pd.concat([total.iloc[:,0], total.iloc[:,2], total.iloc[:,4]])
    b = pd.concat([total.iloc[:,1], total.iloc[:,3], total.iloc[:,5]])
    aa = pd.DataFrame(a).reset_index(drop=True)
    aa.columns = ['idx']
    bb = pd.DataFrame(b).reset_index(drop=True)
    bb.columns = ['signal']
#
    total_x = pd.concat([aa,bb], axis = 1).reset_index(drop=True)
    
    
    total_y = pd.concat([person_1_y, person_2_y, person_3_y]).reset_index(drop= True)
    
    infile_concated_df = pd.concat([total_x, total_y], axis = 1).reset_index(drop= True)
    infile_concated_df.columns = ['person', 'signal', 'talking']
    infile_concated_df = infile_concated_df.reset_index(drop=True)
    
    return infile_concated_df

# 절댓값 취하기
def absolute(df) : 
    absolute_df = np.abs(df)
    return absolute_df

# Train set과 Test set을 각각 구성하는 15개 시나리오, 5개의 시나리오들을 행방향으로 이어주기 위한 함수 
def merge_dataframe(global_data, f_lst): 
    for idx, f in enumerate(f_lst):
        if idx == 0:
            total_df = global_data[f]

        else:
            total_df = total_df.append(global_data[f])
            
    total_df.reset_index(drop=True)
    return total_df