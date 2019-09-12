import preprocess_footnotes_data as pfd
import pandas as pd
import join_pickle_data as jpd
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import numpy as np
from scipy import sparse
df = pd.read_pickle('filter5_4 alternate col to footnotes.pkl')
# 이 단계에서 카테고리로 바꾸는 식으로 데이터 프레임의 크기를 좀 줄여야 한다.

df = df.reset_index(drop=True)
# morpheme = 'kkma'
# morpheme = 'komoran'
# morpheme = 'okt'
morpheme = 'hannanum'

# how_to_pos_treat_as_feature = 'use_only_morph'
how_to_pos_treat_as_feature = 'attach_tag_to_pos'
# how_to_pos_treat_as_feature = 'seperate_tag_and_pos'

pfd.test_pos_for_check(df, morpheme, how_to_pos_treat_as_feature)

path_dir = 'C:/Users/lab515/PycharmProjects/crawlDartFootNote/divide_by_sector'
df = jpd.join_pickle_data(path_dir)
df.to_pickle('filter6 hannanum_attach_tag_to_pos.pkl')

# jpd.join_pickle_data()


# fnguide_fin_ratio_dat = pd.read_excel('main_dependant_var.xlsx', dtype=object)
# file_name = 'filter6 hannanum_attach_tag_to_pos_0.pkl'
# df = pd.read_pickle('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\divide_by_sector\\'+file_name)
# file_name = 'filter6 komoran_attach_tag_to_pos_0.pkl'
# df1 = pd.read_pickle('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\divide_by_sector\\'+file_name)
# df = pfd.merge_fnguide_data_and_filter_no_data(df, fnguide_fin_ratio_dat, file_name)