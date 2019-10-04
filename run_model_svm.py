import preprocess_footnotes_data as pfd
import pandas as pd
# import join_pickle_data as jpd
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import numpy as np
from scipy import sparse


def make_category_by_quartile(df, col_name):
    df = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
                                   , dtype=object, sheet_name='Sheet1')  # for test
    # col_name = 'M수정PER'  # for test
    col_name = 'M수정PER계속사업'  # for test
    cd_list = list(df['Symbol'].unique())
    # per_categorize = []
    for cd in cd_list:
        # cd = cd_list[1]  # for test
        tmp = df[df['Symbol'] == cd]
        tmp = tmp.replace(0, np.nan)  # 이상한 경우.
        q_cat = pd.qcut(tmp[col_name],  [0, .25, .75, 1.], labels=False, duplicates='drop')  # 1이 0~25%, 2가 25%~75%, 3이 75%~100%
        for index, row in tmp.iterrows():
            # df.loc[index, '수정PER3분할'] = q_cat[index]
            df.loc[index, '수정PER계속사업3분할'] = q_cat[index]
    # per_categorize.extend(list(q_cat))
    # df.loc['수정PER3분할'] = per_categorize
    # df = pd.concat([df, pd.concat(per_categorize).rename('수정PER계속사업3분할')], axis=1)
    # df.to_csv('PER_categorization.csv', encoding='cp949')
    df.to_csv('PER_persistent_categorization.csv', encoding='cp949')


def get_real_eps_change(df):  # 전처리 위한 코드. eps 변화율 데이터에 적자로 전환, 흑자 유지 같은 식으로 나와서 수정 eps를 그대로 계산.
    df = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for independent variable\\for_PER_independant_var.xlsx'
                                   , dtype=object, sheet_name='EPS_change_rate_to_calculate')
    # EPS증가율(전분기) :((수정EPS / 수정EPS(-1Q)) - 1) *100
    cd_list = list(df['Symbol'].unique())
    eps_change_ratio = []
    for cd in cd_list:
        # cd = cd_list[3]  # for test
        tmp = df[df['Symbol']== cd]
        eps_change_ratio.extend(list(tmp[tmp.columns[-1]].pct_change()*100))
    df['eps_change_ratio'] = eps_change_ratio
    print(df.shape)
    df = df[(df.회계년 != 2012)]
    print(df.shape)
    df.to_csv('eps_change_ratio.csv', encoding='cp949')


def rearrange_calender_data(df):  # 전처리 위한 코드. eps 변화율 데이터에 적자로 전환, 흑자 유지 같은 식으로 나와서 수정 eps를 그대로 계산.
    df = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for independent variable\\for_PER_independant_var.xlsx'
                                   , dtype=object, sheet_name='return')
    # df = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
    #                                , dtype=object, sheet_name='return')
    # '수정주가수익률', #
    df.drop(columns=['Kind', 'Item', 'Item Name ', 'Frequency'], inplace=True)

    arrange_df = pd.DataFrame(columns=['Symbol', 'Name', '결산월', '회계년', '주기', '수정주가분기수익률'])
    # arrange_df = pd.DataFrame(columns=['Symbol', 'Name', '결산월', '회계년', '주기', '수정주가평균분기'])
    i = 0
    col_names = df.columns[2:]
    for index, row in df.iterrows():
        for col in col_names:
            arrange_df.loc[i] = [row.Symbol, row['Symbol Name'], col.month, col.year, str(int(col.month/3))+"Q", row[col]]
            i += 1
    print(df.shape)
    print(arrange_df.shape)
    # set_diff_df = pd.concat([arrange_df[(arrange_df['회계년']==2019)&(arrange_df['결산월']==9)]
    #                             , arrange_df]).drop_duplicates(keep=False)
    # print(set_diff_df.shape[0]/df.shape[0] == (4*len([2013,2014,2015,2016,2017,2018])+2))
    print(arrange_df.shape[0]/df.shape[0] == (4*len([2013,2014,2015,2016,2017,2018])+2))
    # set_diff_df.to_csv('price.csv', encoding='cp949')
    arrange_df.to_csv('price_rearrange.csv', encoding='cp949')


def match_among_t(df1, df2):
    # df1 = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
    #                                , dtype=object, sheet_name='calc table')
    # df2 = arrange_df
    df1 = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
                                   , dtype=object, sheet_name='Sheet1')
    df2 = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
                                   , dtype=object, sheet_name='Sheet3')
    print(df1.shape)
    print(df2.shape)
    df3 = pd.merge(left=df1, right=df2, how='left', on=['Symbol', 'Name', '회계년', '주기'], sort=False)  # 결산월은 안 맞는 곳이 이따금 있음. 현대해상 2013년 사례가 대표적.
    print(df3.shape)
    df3.to_csv('match.csv', encoding='cp949')

"""
def add_t_minus_dep_var(quanti_ind_var, dep_vars, dep_var, dep_var_t_minus):  # 이전분기 데이터를 반영. 어차피 예상은 t+1에 대한 것이다. # 생각해보니 그냥 t0 시점 PER를 써또 될거 같다. 안써도 될듯하니 제외.
    dep_var = 'M수정PER'
    dep_var_t_minus = 'M수정PERt_minus'
    path_dir = 'C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote'  # done (파일사이즈 문제와 전처리 편의를 위해 pickle로 저장하게 함.)
    quanti_ind_var = pd.read_pickle(path_dir+'\\merged_FnGuide\\quanti_per_predict.pkl')
    dep_vars = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
                             , dtype=object, sheet_name='Sheet1')
    ### for test ###
    identifier = ['Symbol', 'Name', '결산월', '회계년', '주기']
    dep_vars = pd.concat([dep_vars.loc[:, identifier],
                         dep_vars.loc[:, dep_var]], axis=1)  # 일단 폼은 유지하되 필요한 변수 하나만 떼다 쓰기 위함.
    # dep_vars = dep_vars.replace(0, np.nan)  # 가끔 데이터가 없는데 0으로 채워진 경우는 그냥 이렇게 한다. 어차피 데이터가 없을게 뻔해서 의미는 없지만 혹시 모르니까. 특히 거래가 없는 경우. # regression이라면 이 코드가 필요하지만 classification엔 필요 없는 일이었다. 0,1,2로 클래스가 나뉘는 마당에.
    dep_vars.dropna(thresh=6, inplace=True)  # 식별 위한 정보를 제외한 것이 없는 경우. 어차피 이게 없으면 아무것도 안되므로.
    # print(dep_var.info())

    for index, row in quanti_ind_var.iterrows():
        t_minus_quarter = str(int(row['주기'][0])-1)+'Q'   # 결산월 보다 쿼터 쪽이 신뢰도 있음
        if t_minus_quarter == '0Q':  # tplus_quarter가 원래는 4분기라 다음년도 1분기라면
            t_minus_year = int(row['회계년']) - 1   # 결산월 보다 쿼터 쪽이 신뢰도 있음
            t_minus_quarter = '4Q'
        else:
            t_minus_quarter = int(row['회계년'])
        print(t_minus_year)
        print(t_minus_quarter)
        t_minus_data = dep_vars[(dep_vars['Symbol'] == row['Symbol']) &
                                (dep_vars['회계년'] == t_minus_year) &
                                (dep_vars['주기'] == t_minus_quarter)]
        if t_minus_data.shape[0] > 1:  # 중복된 분기의 데이터 없는 문제 확인.
            print('t_minus_data :', t_minus_data)  # 일단 미리 없애놨으니 나타날린 없지만 그래도 체크.
        if t_minus_data.empty:
            quanti_ind_var.loc[index, dep_var] = np.nan
            print('empty', index)
            # print(row)
            continue
        quanti_ind_var.loc[index, dep_var_t_minus] = t_minus_data[dep_var].iloc[0]
    quanti_ind_var.drop(columns=[dep_var_t_minus])
    # df = pd.merge(left=quanti_ind_var, right=dep_vars,
    #               how='left', on=['Symbol', 'Name', '회계년', '주기'], sort=False)  # 결산월은 안 맞는 곳이 이따금 있음. 현대해상 2013년 사례가 대표적.
    # df.to_pickle('./merged_FnGuide/quanti_eps_predict.pkl')
    # df1 = pd.read_pickle('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\merged_FnGuide\\quanti_eps_predict.pkl')
"""


if __name__ == '__main__':  # 시간내로 하기 위해 멀티프로세싱 적극 활용 요함.
    # 최초엔 일단 종속_리스트, 독립변수명을 정해두는것이 편해보인다.
    path_dir = 'C:/Users/lab515/PycharmProjects/crawlDartFootNote'  # done (파일사이즈 문제와 전처리 편의를 위해 pickle로 저장하게 함.)
    dep_var = '수정PER3분할'
    ind_var_list = ['M000701061_수정PBR(배)', 'M000901001_ln총자산(천원)', 'debt_asset_ratio', 'eps_change_ratio', '수정주가분기수익률']
    quanti_data_set_file_name = '/merged_FnGuide/quanti_per_predict.pkl'
    qual_set_file_name = '/divide_by_sector/filter7 hannanum_filtered_pos.pkl'
    quanti_qual_matched_file_name = 'quanti_qaul_eps_predict.pkl'
    """
    # fnguide_fin_ratio_dat = pd.read_excel('main_dependant_var.xlsx', dtype=object)
    # file_name = 'filter6 komoran_attach_tag_to_pos_0.pkl'
    # df = pd.read_pickle('C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\divide_by_sector\\'+file_name)
    # df = pfd.merge_fnguide_data_and_filter_no_data(df, fnguide_fin_ratio_dat, file_name)

    quanti_ind_var = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for independent variable\\for_PER_independant_var.xlsx'
                                   , dtype=object, sheet_name='Sheet1')
    dep_vars = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
                            , dtype=object, sheet_name='Sheet1')
    data_set_file_name = 'quanti_per_predict.pkl'
    # matched_quanti_data = pfd.match_fnguide_data_among_them(quanti_ind_var, dep_var, '수정EPS\(원\)', data_set_file_name)  # 뒤에 붙이는 키워드는 정말 예측하고 싶은 종속변수명을 특정하기 위함.
    matched_quanti_data = pfd.match_fnguide_data_among_them(quanti_ind_var, dep_vars, dep_var, ind_var_list, data_set_file_name)  # 뒤에 붙이는 키워드는 정말 예측하고 싶은 종속변수명을 특정하기 위함.
    """
    quanti_ind_var = pd.read_pickle(path_dir + quanti_data_set_file_name)
    # ['Symbol', 'Name', '결산월', '회계년', '주기', 'M000701061_수정PBR(배)', 'M000901001_ln총자산(천원)', 'debt_asset_ratio', 'eps_change_ratio', '수정주가분기수익률', '수정PER3분할']
    qual_ind_var = pd.read_pickle(path_dir + qual_set_file_name)
    # qual_ind_var_after = filter_number(qual_ind_var)  # 앞으로 숫자제거는 필수로.
    # qual_ind_var_after.to_pickle(path_dir + quanti_data_set_file_name)

    # 실제 있는 열의 목록 columns=['crp_cd', 'ind_cd', 'crp_cls', 'crp_nm', 'rpt_nm', 'foot_note', 'rcp_dt']
    # qual_ind_var = add_one_hot(qual_ind_var, 'crp_cls')  # for test
    # qual_ind_var = add_one_hot_with_ind_cd(qual_ind_var)  # for test
    qual_ind_var = pfd.add_one_hot(qual_ind_var, 'crp_cls')
    qual_ind_var.drop(['crp_nm', 'rcp_dt'], axis=1, inplace=True)

    matched_quanti_and_qual_data = match_fnguide_data_and_delete_redundant(qual_ind_var, quanti_ind_var,
                                                                           'quanti_qaul_eps_predict.pkl')

    # matched_quanti_and_qual_data = pd.merge(left=quanti_ind_var, right=qual_ind_var, how='left', on=['Symbol', 'Name', '회계년', '주기'], sort=False)
    # matched_quanti_and_qual_data = matched_quanti_and_qual_data.drop(columns=['acc_crp'])

    # columns = ['crp_nm', 'rcp_dt']  # 결합했으므로 불필요한 열. 'ind_cd', 'crp_cls'는 one-hot으로 바꾸는 과정에서 제거됨.
    # qual_ind_var.drop(columns, inplace=True, axis=1)



    matched_quanti_and_qual_data = pd.merge

    matched_quanti_and_qual_data, valid_df_idx_list = pfd.match_fnguide_data_and_delete_redundant(qual_ind_var, matched_quanti_data, data_set_file_name)

    dataset, y = pfd.filter_pos(valid_df_idx_list, matched_quanti_and_qual_data)

    """
    # matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/quanti_qaul_eps_predict.pkl')
    

    start_time = datetime.now()
    print("start_time : ", start_time)
    rms_list1 = pfd.previous_research_with_svm(matched_quanti_data.values, 30)
    print("take time : {}".format(datetime.now() - start_time))
    """
    """
    # matched_quanti_and_qual_data = pd.read_pickle('merged_FnGuide/quanti_qaul_eps_predict.pkl')
    X = sparse.load_npz('./merged_FnGuide/dataset.npz')
    matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/quanti_qaul_eps_predict.pkl')
    y = matched_quanti_and_qual_data.values[:, -1]

    start_time = datetime.now()
    print("start_time : ", start_time)
    rms_list2 = pfd.svm_with_foot_note(X, y, 30)
    # rms_list2 = svm_with_foot_note(X, y, 30)
    print("take time : {}".format(datetime.now() - start_time))

    # pfd.equ_var_test_and_unpaired_t_test(rms_list1, rms_list2)  # 독립 t-test
    """
