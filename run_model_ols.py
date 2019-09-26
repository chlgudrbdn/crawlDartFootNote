import preprocess_footnotes_data as pfd
import pandas as pd
import join_pickle_data as jpd
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from calendar import monthrange
from numpy import dot
from numpy.linalg import norm
import numpy as np
from scipy import sparse
import statsmodels.api as sm
import gzip
import pickle
import multiprocessing


def cos_sim(A, B):
    return dot(A, B)/(norm(A)*norm(B))


def ols(data, x_var_names, y_var_name):
    X = data[x_var_names]
    y = data[y_var_name]
    X = sm.add_constant(X)
    result = sm.OLS(np.asarray(y), X).fit()
    print(result.rsquared)
    # 요약결과 출력
    print(result.summary())
    # 요약결과 저장
    f = open("OLS_result.txt", 'w')
    f.write(result.summary())
    f.close()


# def footnote_to_string(index):
#     df6.loc[index, 'foot_note'] = ''.join(df6.loc[index, 'foot_note'])


# def footnote_to_list(index):
    # for index, row in df.iterrows():
    # li = df6.loc[index, 'foot_note'].split(';')
    # df6.loc[index, 'foot_note'] = [s + ';' for s in df6.loc[index, 'foot_note'].split(';')]


if __name__ == '__main__':  # 시간내로 하기 위해 멀티프로세싱 적극 활용 요함.
    path_dir = 'C:/Users/lab515/PycharmProjects/crawlDartFootNote/divide_by_sector'
    # df6 = pd.read_pickle(path_dir + '/filter6 hannanum_attach_tag_to_pos.pkl')
    df6 = jpd.join_pickle_data(path_dir, 'hannanum')
    # df6 = pd.read_csv(path_dir+'/filter6 hannanum_attach_tag_to_pos.csv')
    # df6.to_pickle(path_dir+'/filter6 hannanum_attach_tag_to_pos.pkl')
    # df6.to_csv(path_dir+'/filter6 hannanum_attach_tag_to_pos.csv')

    # 멀티프로세싱 해봤는데 메모리 에러만 생긴다.
    # pool = multiprocessing.Pool(processes=4)  # 현재 시스템에서 사용 할 프로세스 개수
    # pool.map(footnote_to_list, list(df6.index))
    # pool.close()
    # pool.join()
    # for index, row in df6.iterrows():
    #     df6.loc[index, 'foot_note'] = [s + ';' for s in df6.loc[index, 'foot_note'].split(';')]
    df7 = pfd.filter_pos(df6, ['N;', 'P;', 'F;'])  # 한나눔은 체언, 용언, 외국어 말곤 딱히 쓸모있는 태그 분류는 하지 않는 것으로 보인다.
    del df6
    df7.to_pickle(path_dir + '/filter7 hannanum_filtered_pos.pkl')
    """

    pre_crp_cd = ""
    pre_rpt_nm = ""
    # pre_rpt_dt = ""
    pre_index = ""
    index_list_for_delete = []
    for index, row in df6.iterrows():
        cur_crp_cd = row['crp_cd']
        cur_rpt_nm = row['rpt_nm']
        cur_rcp_dt = row['rcp_dt']
        if row['dic_cls'] != 'NA':
            if cur_crp_cd == pre_crp_cd and cur_rpt_nm == pre_rpt_nm:
                t_closing_date = cur_rpt_nm[cur_rpt_nm.find("(") + 1:cur_rpt_nm.find(")")].split('.')
                tplus_closing_date = datetime(int(t_closing_date[0]), int(t_closing_date[1]), monthrange(int(t_closing_date[0]), int(t_closing_date[1]))[
                                                  1]) + relativedelta(months=3)

                cur_rcp_dt_li = cur_rcp_dt.split('.')
                rpt_dt_datetime = datetime(int(cur_rcp_dt_li[0]), int(cur_rcp_dt_li[1]), int(cur_rcp_dt_li[2]))
                # if cur_rpt_nm[:5] == '사업보고서':
                #     index_list_for_delete.append(pre_index)  # 사업 보고서의 경우 3개월 보다 더 오래 걸리기도 하는 모양
                # else:
                if rpt_dt_datetime <= tplus_closing_date:  # 일단 데이터 누수 막기 위해 다음 분기 되기 전 데이터 한정
                    # 일단 당장 지울 필요는 없지만 일단은 중복이 있다는 얘기. 즉 대체가 필요하다는 소리
                    index_list_for_delete.append(pre_index)  # 즉 중복됐으면서 보다 이전날짜에 나온 공시를 지워야함.
                else:  # 만약 다음 분기 이후에 정정되었다면 일단 이건 쓸모가 없단 소리.
                    index_list_for_delete.append(index)
        pre_crp_cd = cur_crp_cd
        pre_rpt_nm = cur_rpt_nm
        # pre_rpt_dt = cur_rcp_dt
        pre_index = index
    print(index_list_for_delete[0:5])

    df4_2 = df3_2.drop(index_list_for_delete)
    print(df3_2.shape)
    print(df4_2.shape)

    df4_2.to_pickle('filter4_2 with_revision.pkl')



    fnguide_fin_ratio_dat = pd.read_excel('main_dependant_var.xlsx', dtype=object)
    file_name = 'filter6 komoran_attach_tag_to_pos_6.pkl'  # for test
    df = pd.read_pickle('C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\divide_by_sector\\'+file_name)
    df = pfd.merge_fnguide_data_and_filter_no_data(df, fnguide_fin_ratio_dat, file_name)

    quanti_ind_var = pd.read_excel('C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\previous research independant variable\\about_EPS_independant_var.xlsx'
                                   , dtype=object, sheet_name='Sheet1')
    dep_var = pd.read_excel('C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable retrived 2019_05_15\\EPS_rawData.xlsx'
                                   , dtype=object, sheet_name='Sheet1')
    data_set_file_name = 'quanti_eps_predict.pkl'
    # matched_quanti_data = pfd.match_fnguide_data_among_them(quanti_ind_var, dep_var, '수정EPS\(원\)', data_set_file_name)  # 뒤에 붙이는 키워드는 정말 예측하고 싶은 종속변수명을 특정하기 위함.
    matched_quanti_data = pfd.match_fnguide_data_among_them(quanti_ind_var, dep_var, '수정EPS\(계속사업\)\(원\)', data_set_file_name)  # 뒤에 붙이는 키워드는 정말 예측하고 싶은 종속변수명을 특정하기 위함.

    qual_ind_var = pd.read_pickle('filter5_2 alternate col to footnotes.pkl')
    """
    """
    # dataset = sparse.load_npz('./merged_FnGuide/qual_matix.npz')
    path_dir = 'C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\divide_by_sector'  # done (파일사이즈 문제와 전처리 편의를 위해 pickle로 저장하게 함.)
    qual_ind_var = pd.read_pickle(path_dir + '\\filter6 komoran_attach_tag_to_pos.pkl')
    columns = ['ind_cd', 'crp_cls', 'crp_nm', 'foot_note', 'rcp_dt']  # columns=['crp_cd', 'ind_cd', 'crp_cls', 'crp_nm', 'rpt_nm', 'foot_note', 'rcp_dt']
    qual_ind_var.drop(columns, inplace=True, axis=1)

    # dataset = sparse.load_npz('./merged_FnGuide/qual_matix.npz')

    data_set_file_name = 'quanti_qaul_eps_predict.pkl'
    matched_quanti_data = pd.read_pickle('merged_FnGuide/quanti_eps_predict.pkl')

    matched_quanti_and_qual_data, valid_df_idx_list = pfd.match_fnguide_data_and_delete_redundant(qual_ind_var, matched_quanti_data, data_set_file_name)

    dataset, y = pfd.filter_nouns_already_tagged(valid_df_idx_list, matched_quanti_and_qual_data)
    """
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
