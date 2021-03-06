import pandas as pd
# import join_pickle_data as jpd
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import numpy as np
from scipy import sparse
from scipy.sparse import csr_matrix
import os
import preprocess_footnotes_data as pfd


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
    # df = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for independent variable\\for_PER_independant_var.xlsx'
                                   # , dtype=object, sheet_name='EPS_change_rate_to_calculate')
    df = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
                                   , dtype=object, sheet_name='return_rearrange_from2012')  # 'price_rearrange.csv'
                                   # , dtype=object, sheet_name='return_from2012')
    # EPS증가율(전분기) :((수정EPS / 수정EPS(-1Q)) - 1) *100
    cd_list = list(df['Symbol'].unique())
    eps_change_ratio = []
    for cd in cd_list:
        # cd = cd_list[3]  # for test
        tmp = df[df['Symbol'] == cd]
        eps_change_ratio.extend(list(tmp[tmp.columns[-1]].pct_change()*100))
    df['eps_change_ratio'] = eps_change_ratio
    print(df.shape)
    df = df[(df.회계년 != 2012)]
    print(df.shape)
    df.to_csv('eps_change_ratio.csv', encoding='cp949')


def rearrange_calender_data(df):  # 전처리 위한 코드. eps 변화율 데이터에 적자로 전환, 흑자 유지 같은 식으로 나와서 수정 eps를 그대로 계산.
    df = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for independent variable\\for_PER_independant_var.xlsx'
                                   , dtype=object, sheet_name='return')
    # '수정주가수익률', #
    df.drop(columns=['Kind', 'Item', 'Item Name ', 'Frequency'], inplace=True)

    # arrange_df = pd.DataFrame(columns=['Symbol', 'Name', '결산월', '회계년', '주기', '수정주가분기수익률'])
    arrange_df = pd.DataFrame(columns=['Symbol', 'Name', '결산월', '회계년', '주기', '수정주가평균분기'])
    i = 0
    col_names = df.columns[2:]  # 행에 회사 열에 분기말 날짜.
    for index, row in df.iterrows():
        for col in col_names:  # 한 줄 내려가고 오른쪽으로 이동하며 하나씩 쌓는다.
            arrange_df.loc[i] = [row.Symbol, row['Symbol Name'], col.month, col.year, str(int(col.month/3))+"Q", row[col]]
            i += 1
    print(df.shape)
    print(arrange_df.shape)
    # set_diff_df = pd.concat([arrange_df[(arrange_df['회계년']==2019)&(arrange_df['결산월']==9)]
    #                             , arrange_df]).drop_duplicates(keep=False)
    # print(set_diff_df.shape[0]/df.shape[0] == (4*len([2013,2014,2015,2016,2017,2018])+2))
    print(arrange_df.shape[0]/df.shape[0] == (4*len([2013,2014,2015,2016,2017,2018])+3))
    # set_diff_df.to_csv('price.csv', encoding='cp949')
    arrange_df.to_csv('price_rearrange.csv', encoding='cp949')


def match_among_t(df1, df2):
    # df1 = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
    #                                , dtype=object, sheet_name='calc table')
    # df2 = arrange_df
    df1 = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
                                   , dtype=object, sheet_name='Sheet1')
    df2 = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
                                   , dtype=object, sheet_name='rearranged_from2003')
    df2 = pd.read_csv('PER_categorization_with_1q_20.csv', encoding='cp949')
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



def match_fnguide_data_among_them(quanti_ind_var, dep_vars, dep_var, ind_var_list, file_name):
    # quanti_ind_var = pd.read_excel('C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\previous research independant variable\\about_EPS_independant_var.xlsx', dtype=object, sheet_name='Sheet1')
    # dep_var = pd.read_excel('C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable retrived 2019_05_15\\EPS_rawData.xlsx', dtype=object, sheet_name='Sheet1')
    # keyword = '수정EPS\(원\)'
    # file_name = ''
    identifier = ['Symbol', 'Name', '결산월', '회계년', '주기']
    dep_vars = pd.concat([dep_vars.loc[:, identifier],
                         dep_vars.loc[:, dep_var]], axis=1)  # 일단 폼은 유지하되 필요한 변수 하나만 떼다 쓰기 위함.
    # dep_vars = dep_vars.replace(0, np.nan)  # 가끔 데이터가 없는데 0으로 채워진 경우는 그냥 이렇게 한다. 어차피 데이터가 없을게 뻔해서 의미는 없지만 혹시 모르니까. 특히 거래가 없는 경우. # regression이라면 이 코드가 필요하지만 classification엔 필요 없는 일이었다. 0,1,2로 클래스가 나뉘는 마당에.
    dep_vars.dropna(thresh=6, inplace=True)  # 식별 위한 정보를 제외한 것이 없는 경우. 어차피 이게 없으면 아무것도 안되므로.
    # print(dep_var.info())

    # print(quanti_dep_var.columns)
    # print('ind_var : ', ind_var)
    identifier.extend(ind_var_list)
    # print('new ind_var : ', identifier)
    # quanti_ind_var = quanti_ind_var.drop(columns=list(quanti_ind_var.loc[:, quanti_ind_var.columns.str.contains('^주기')].columns))  # 주기는 불필요하니 제거
    quanti_ind_var = quanti_ind_var[identifier]
    print(quanti_ind_var.shape)
    quanti_ind_var = quanti_ind_var.replace(0, np.nan)  # 가끔 데이터가 없는데 0으로 채워진 경우는 그냥 이렇게 한다. 어차피 데이터가 없을게 뻔해서 의미는 없지만 혹시 모르니까.
    quanti_ind_var.dropna(thresh=len(identifier), inplace=True)  # 식별 위한 정보를 제외한 것이 없는 경우. 어차피 이게 없으면 아무것도 안되므로.
    # print(quanti_ind_var.columns)
    print('quanti_ind_var.info() : ', quanti_ind_var.shape)
    # print(dep_var)
    # result_df = pd.DataFrame()
    for index, row in quanti_ind_var.iterrows():
        # print(index)
        tplus_quarter = str(int(row['주기'][0])+1)+'Q'   # 결산월 보다 쿼터 쪽이 신뢰도 있음
        if tplus_quarter == '5Q':  # tplus_quarter가 원래는 4분기라 다음년도 1분기라면
            tplus_year = int(row['회계년']) + 1   # 결산월 보다 쿼터 쪽이 신뢰도 있음
            tplus_quarter = '1Q'
        else:
            tplus_year = int(row['회계년'])
        tplus_data = dep_vars[(dep_vars['Symbol'] == row['Symbol'])
                              & (dep_vars['회계년'] == tplus_year)
                              & (dep_vars['주기'] == tplus_quarter)]
        if tplus_data.shape[0] > 1:  # 중복된 분기의 데이터 없는 문제 확인.
            print('tplus_data :', tplus_data)  # 일단 미리 없애놨으니 나타날린 없지만 그래도 체크.
        if tplus_data.empty:
            # result_df = result_df.append(pd.Series(), ignore_index=True)  # 매칭하는 날짜는 있는데 비어있는 경우 일단 빈칸으로 채우고 넘어간다.
            quanti_ind_var.loc[index, dep_var] = np.nan
            print('empty', index)
            # print(row)
            continue
        # result_df = result_df.append(tplus_data, ignore_index=True)  # 찾은 결과를 한줄씩 붙인뒤 나중에 옆으로 붙일 예정.
        # print('tplus_data : ', tplus_data.values)
        # print('tplus_data : ', tplus_data[dep_var].iloc[0])
        # print('tplus_data : ', tplus_data.loc[:, dep_var].values[0].tolist())
        # print('tplus_data : ', int(tplus_data.loc[:, '수정PER3분할_10y_20p']))
        for dep in dep_var:
            quanti_ind_var.loc[index, dep] = int(tplus_data.loc[:, dep])  # t+1 분기와 매칭
    # result_df = result_df.drop(columns=['Symbol', 'Name', '결산월', '회계년'])  # 종속변수 쪽 식별 정보는 필요 없음.
    # result_df.reset_index(drop=True, inplace=True)
    # quanti_ind_var.reset_index(drop=True, inplace=True)
    # quanti_ind_var = quanti_ind_var.drop(columns=['Name'])  # 그냥 종목번호보다 보기좋아서 냅둔거라. 지워도 상관없음.

    # quanti_ind_var = pd.concat([quanti_ind_var, result_df], axis=1)
    quanti_ind_var.dropna(inplace=True)  # 이전 단계에서 걸렀을 가능성이 높지만 그래도 1~2개 없는 경우를 거르기 위함.
    # directory_name = 'merged_FnGuide ind_var/'
    # if not os.path.exists(directory_name):
    #     os.mkdir(directory_name)
    # df1.to_excel(directory_name+'/merged_FnGuide ind_var '+keyword+'.xlsx')
    directory_name = './merged_FnGuide'
    if not os.path.exists(directory_name):  # bitcoin_per_date 폴더에 저장되도록, 폴더가 없으면 만들도록함.
        os.mkdir(directory_name)
    # np.save('./merged_FnGuide/'+file_name, quanti_ind_var.values)
    print(quanti_ind_var.iloc[:, 5:].astype('float'))
    quanti_ind_var.to_pickle(directory_name+'/'+file_name)
    print(quanti_ind_var.columns)
    print(quanti_ind_var.shape)
    print(quanti_ind_var.info())

    return quanti_ind_var  # 다음에 쓰려고 ndarray로 반환하지 않음.


if __name__ == '__main__':  # 시간내로 하기 위해 멀티프로세싱 적극 활용 요함.
    # 최초엔 일단 종속_리스트, 독립변수명을 정해두는것이 편해보인다.
    path_dir = 'C:/Users/lab515/PycharmProjects/crawlDartFootNote'  # done (파일사이즈 문제와 전처리 편의를 위해 pickle로 저장하게 함.)

    # dep_var = '수정PER3분할_10y_20p'  # 수정PER3분할_10y_20p, 수정PER3분할_10y_25p, 수정PER3분할_10y_33p, 수정PER3분할_1q_20p, 수정PER3분할_1q_25p, 수정PER3분할_1q_33p
    dep_var = ['수정PER3분할_10y_20p', '수정PER3분할_10y_25p', '수정PER3분할_10y_33p', '수정PER3분할_1q_20p', '수정PER3분할_1q_25p', '수정PER3분할_1q_33p']

    ind_var_list = ['M000701061_수정PBR(배)', 'M000901001_ln총자산(천원)', 'debt_asset_ratio', 'eps_change_ratio', '수정주가분기수익률']
    quanti_data_set_file_name = 'quanti_per_komoran.pkl'
    quanti_qual_matched_file_name = 'quanti_qaul_komoran.pkl'

    """
    dep_vars = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
                            , dtype=object, sheet_name='Sheet1')  # 종속변수
    valid = pd.read_pickle(path_dir + '/divide_by_sector/filter8 komoran_for_cosine_distance.pkl')  # 독립변수중 산업이나 코사인거리
    quanti_ind_var = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for independent variable\\for_PER_independant_var.xlsx'
                                   , dtype=object, sheet_name='Sheet1')

    matched_quanti_data = match_fnguide_data_among_them(quanti_ind_var, dep_vars, dep_var, ind_var_list, quanti_data_set_file_name)

    dep_vars = dep_vars[['Symbol', 'Name', '결산월', '회계년', '주기', 'M수정PER']]
    dep_vars.columns = ['Symbol', 'Name', '결산월', '회계년', '주기', 'M수정PER_tminus']
    matched_quanti_data = pd.merge(left=matched_quanti_data, right=dep_vars, how='left', on=['Symbol', 'Name', '회계년', '주기'], sort=False)  # 결산월은 안 맞는 곳이 이따금 있음. 현대해상 2013년 사례가 대표적.

    matched_quanti_and_qual_data, valid_df_idx_list = pfd.match_quanti_and_qual_data(valid, matched_quanti_data, quanti_qual_matched_file_name)
    matched_quanti_and_qual_data = pfd.add_one_hot(matched_quanti_and_qual_data, 'crp_cls')
    matched_quanti_and_qual_data = pfd.add_one_hot(matched_quanti_and_qual_data, '주기')
    matched_quanti_and_qual_data = pfd.add_one_hot_with_ind_cd(matched_quanti_and_qual_data)

    columns = ['crp_cd', 'crp_nm', 'rpt_nm', 'foot_note', 'rcp_dt', 't_minus_index',
               't_minus_year_index', 'Symbol', 'Name', '결산월_x', '결산월_y', '회계년']  # 주기는 일단 남긴다.
    matched_quanti_and_qual_data.drop(columns, inplace=True, axis=1)
    matched_quanti_and_qual_data.dropna(inplace=True)
    matched_quanti_and_qual_data.to_pickle(path_dir + '/merged_FnGuide/' + quanti_qual_matched_file_name)


    # matched_quanti_and_qual_data.drop(['t_1y_cos_dist'], inplace=True, axis=1)
    # matched_quanti_and_qual_data.drop(['t_1q_cos_dist'], inplace=True, axis=1)
    # matched_quanti_and_qual_data = matched_quanti_and_qual_data[matched_quanti_and_qual_data.t_1q_cos_dist != ""]
    # matched_quanti_and_qual_data['t_1q_cos_dist'] = matched_quanti_and_qual_data['t_1q_cos_dist'].astype('float')
    """

    """
    # fnguide_fin_ratio_dat = pd.read_excel('main_dependant_var.xlsx', dtype=object)
    # file_name = 'filter6 komoran_attach_tag_to_pos_0.pkl'
    # df = pd.read_pickle('C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\divide_by_sector\\'+file_name)
    # df = pfd.merge_fnguide_data_and_filter_no_data(df, fnguide_fin_ratio_dat, file_name)

    quanti_ind_var = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for independent variable\\for_PER_independant_var.xlsx'
                                   , dtype=object, sheet_name='Sheet1')
    dep_vars = pd.read_excel('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable\\PER_rawData.xlsx'
                            , dtype=object, sheet_name='Sheet1')
    data_set_file_name = 'quanti_per_hannanum.pkl'
    # matched_quanti_data = pfd.match_fnguide_data_among_them(quanti_ind_var, dep_var, '수정EPS\(원\)', data_set_file_name)  # 뒤에 붙이는 키워드는 정말 예측하고 싶은 종속변수명을 특정하기 위함.
    matched_quanti_data = pfd.match_fnguide_data_among_them(quanti_ind_var, dep_vars, dep_var, ind_var_list, data_set_file_name)  # 뒤에 붙이는 키워드는 정말 예측하고 싶은 종속변수명을 특정하기 위함.
    """
    """
    quanti_ind_var = pd.read_pickle(path_dir + quanti_data_set_file_name)
    # ['Symbol', 'Name', '결산월', '회계년', '주기', 'M000701061_수정PBR(배)', 'M000901001_ln총자산(천원)', 'debt_asset_ratio', 'eps_change_ratio', '수정주가분기수익률', '수정PER3분할']
    qual_ind_var = pd.read_pickle(path_dir + '/divide_by_sector/filter7 komoran_filtered_pos.pkl')
    # qual_ind_var_after = filter_number(qual_ind_var)  # 앞으로 숫자제거는 필수로.
    # qual_ind_var_after.to_pickle(path_dir + quanti_data_set_file_name)

    # 실제 있는 열의 목록 columns=['crp_cd', 'ind_cd', 'crp_cls', 'crp_nm', 'rpt_nm', 'foot_note', 'rcp_dt']
    # qual_ind_var = add_one_hot(qual_ind_var, 'crp_cls')  # for test
    # qual_ind_var = add_one_hot_with_ind_cd(qual_ind_var)  # for test
    # qual_ind_var = pfd.add_one_hot(qual_ind_var, 'crp_cls')  # 이미 다해뒀다.

    matched_quanti_and_qual_data, valid_df_idx_list = pfd.match_quanti_and_qual_data(qual_ind_var, quanti_ind_var,
                                                                  'quanti_qaul_per_komoran.pkl')
    columns = ['crp_cd', 'rpt_nm', 'Symbol', 'Name', '결산월_x', '결산월_y', '회계년', 'crp_nm 'rcp_dt']  # 주기는 일단 남긴다. 년도는 미래 예측에도 강건한 성능을 보이기 위해 빼더라도.
    # columns = ['crp_cd', 'ind_cd', 'crp_cls', 'crp_nm', 'rpt_nm', 'rcp_no', 'dic_cls', 'dcm_no', 'col_dcm_no', 'consolidated_foot_note', 'rcp_dt',
    #            'Symbol', '결산월', '회계년']
    # df5_2의 컬럼 목록 ['crp_cd', 'ind_cd', 'crp_cls', 'crp_nm', 'rpt_nm', 'rcp_no', 'dic_cls', 'dcm_no', 'col_dcm_no', 'foot_note', 'consolidated_foot_note', 'rcp_dt']
    # 현시점에서 산업코드 나누는건 의미 없고(사실 절반이 제조업이라 더더욱), 종목코드는 중복이라 Symbol 삭제,
    # rcp_dt는 이전 단계에 써먹어야 했음, crp_nm은 어차피 종목코드로 대체(검색 편의를 위해 남겼을 뿐),
    # 결산월과 회계년은 이미 t+1과 t0를 맞추는데 사용.
    # rpt_nm은 좀 애매한데 일단 분기 보고서인지 반기보고서인지 나눠서 제어할 필요가 있다고 보고 남김.
    matched_quanti_and_qual_data.dropna(inplace=True)  # 사실 별 의미 없는 짓이다.
    matched_quanti_and_qual_data.drop(columns, inplace=True, axis=1)

    matched_quanti_and_qual_data = pfd.add_one_hot(matched_quanti_and_qual_data, '주기')
    # matched_quanti_and_qual_data = add_one_hot(matched_quanti_and_qual_data, '주기')  # for test
    matched_quanti_and_qual_data.to_pickle(path_dir+'/merged_FnGuide/'+quanti_qual_matched_file_name)

    # matched_quanti_and_qual_data = pd.merge(left=quanti_ind_var, right=qual_ind_var, how='left', on=['Symbol', 'Name', '회계년', '주기'], sort=False)
    # matched_quanti_and_qual_data = matched_quanti_and_qual_data.drop(columns=['acc_crp'])

    # columns = ['crp_nm', 'rcp_dt']  # 결합했으므로 불필요한 열. 'ind_cd', 'crp_cls'는 one-hot으로 바꾸는 과정에서 제거됨.
    # qual_ind_var.drop(columns, inplace=True, axis=1)

    """
    matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/'+quanti_qual_matched_file_name)

    main_ind_var = 't_1q_cos_dist'
    # main_ind_var = 't_1y_cos_dist'

    dep = dep_var[0]  # ['수정PER3분할_10y_20p', '수정PER3분할_10y_25p', '수정PER3분할_10y_33p', '수정PER3분할_1q_20p', '수정PER3분할_1q_25p', '수정PER3분할_1q_33p'] 중 택1
    # dep = dep_var[1]  # ['수정PER3분할_10y_20p', '수정PER3분할_10y_25p', '수정PER3분할_10y_33p', '수정PER3분할_1q_20p', '수정PER3분할_1q_25p', '수정PER3분할_1q_33p'] 중 택1
    # dep = dep_var[2]  # ['수정PER3분할_10y_20p', '수정PER3분할_10y_25p', '수정PER3분할_10y_33p', '수정PER3분할_1q_20p', '수정PER3분할_1q_25p', '수정PER3분할_1q_33p'] 중 택1
    # dep = dep_var[3]  # ['수정PER3분할_10y_20p', '수정PER3분할_10y_25p', '수정PER3분할_10y_33p', '수정PER3분할_1q_20p', '수정PER3분할_1q_25p', '수정PER3분할_1q_33p'] 중 택1
    # dep = dep_var[4]  # ['수정PER3분할_10y_20p', '수정PER3분할_10y_25p', '수정PER3분할_10y_33p', '수정PER3분할_1q_20p', '수정PER3분할_1q_25p', '수정PER3분할_1q_33p'] 중 택1
    # dep = dep_var[5]  # ['수정PER3분할_10y_20p', '수정PER3분할_10y_25p', '수정PER3분할_10y_33p', '수정PER3분할_1q_20p', '수정PER3분할_1q_25p', '수정PER3분할_1q_33p'] 중 택1

    matched_quanti_and_qual_data = matched_quanti_and_qual_data[matched_quanti_and_qual_data[main_ind_var] != ""]  # 택1 결측치 제거.

    matched_quanti_and_qual_data[main_ind_var] = matched_quanti_and_qual_data[main_ind_var].astype('float')
    matched_quanti_and_qual_data[dep_var] = matched_quanti_and_qual_data[dep_var].astype('int8')

    print(matched_quanti_and_qual_data.info())
    cols = list(matched_quanti_and_qual_data.columns)
    if main_ind_var == 't_1q_cos_dist':
        cols.remove('t_1y_cos_dist')
    else:
        cols.remove('t_1q_cos_dist')

    for d_var in dep_var:
        cols.remove(d_var)
    cols.append(dep)
    matched_quanti_and_qual_data = matched_quanti_and_qual_data[cols]

    quanti_data_predict = matched_quanti_and_qual_data.loc[:, matched_quanti_and_qual_data.columns != main_ind_var]

    start_time = datetime.now()
    print("start_time : ", start_time)
    # rms_list1 = previous_research_with_svm(quanti_data_predict.values, 30)
    print('dependent variable: ', dep)
    print('main independent variable: ', main_ind_var)

    acc_list1, f1_list1 = pfd.svm_with_foot_note(quanti_data_predict.iloc[:, :-1].values,
                                                 quanti_data_predict.iloc[:, -1].values, 30)
    print("take time : {}".format(datetime.now() - start_time))
    # acc_list1 = [0.459472,0.459046,0.458718,0.459158,0.458742,0.458874,0.458742,0.458576,0.458782,0.458742,0.458726,0.45907,0.45916,0.459168,0.45921,0.458784,0.45898,0.45898,0.459152,0.458972,0.459134,0.458308,0.459234,0.458642,0.45888,0.459226,0.459342,0.458872,0.458962,0.458732]
    # f1_list1 = [0.226717177,0.2251687,0.227156287,0.226409794,0.2269375,0.226362649,0.227054485,0.22680026,0.227619277,0.226738998,0.22763236,0.227521745,0.22727243,0.226296926,0.22657343,0.226313087,0.227119704,0.226576077,0.226087876,0.227655016,0.226092129,0.225271832,0.227221064,0.225434696,0.227670123,0.22678054,0.22680888,0.227290038,0.227021287,0.227555999]
    # matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/quanti_qaul_per_dataset.pkl')

    # X = sparse.load_npz('./merged_FnGuide/for_per_qual_tf_idf_komoran.npz')
    # X = sparse.load_npz('./merged_FnGuide/for_per_qual_tf_idf_komoran.npz')
    # matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/quanti_qaul_eps_predict.pkl')
    # y = quanti_data_predict.values[:, -1].astype('int8')
    # del quanti_data_predict
    # X = csr_matrix(X)
    start_time = datetime.now()
    print("total start_time : ", start_time)

    # np.random.seed(42)  # for placebo
    # matched_quanti_and_qual_data[main_ind_var] = np.random.rand(matched_quanti_and_qual_data.shape[0])  # for placebo

    acc_list2, f1_list2 = pfd.svm_with_foot_note(matched_quanti_and_qual_data.iloc[:, :-1].values,
                                                 matched_quanti_and_qual_data.iloc[:, -1].values, 30)
    # rms_list2 = svm_with_foot_note(X, y, 30)
    print("total take time : {}".format(datetime.now() - start_time))

    result1 = pfd.equ_var_test_and_unpaired_t_test(acc_list1, acc_list2)  # 독립 t-test 단방향 검정
    result2 = pfd.equ_var_test_and_unpaired_t_test(f1_list1, f1_list2)  # 독립 t-test 단방향 검정
    """
    """