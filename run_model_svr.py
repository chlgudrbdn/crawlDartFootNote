import preprocess_footnotes_data as pfd
import pandas as pd
# import join_pickle_data as jpd
from sklearn.model_selection import train_test_split
from datetime import datetime, timedelta
import numpy as np
from scipy import sparse


if __name__ == '__main__':  # 시간내로 하기 위해 멀티프로세싱 적극 활용 요함.
    # fnguide_fin_ratio_dat = pd.read_excel('main_dependant_var.xlsx', dtype=object)
    # file_name = 'filter6 komoran_attach_tag_to_pos_0.pkl'
    # df = pd.read_pickle('C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\divide_by_sector\\'+file_name)
    # df = pfd.merge_fnguide_data_and_filter_no_data(df, fnguide_fin_ratio_dat, file_name)

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
    path_dir = 'C:\\Users\\lab515\\PycharmProjects\\crawlDartFootNote\\divide_by_sector'  # done (파일사이즈 문제와 전처리 편의를 위해 pickle로 저장하게 함.)
    qual_ind_var = pd.read_pickle(path_dir + '\\filter6 komoran_attach_tag_to_pos.pkl')
    columns = ['ind_cd', 'crp_cls', 'crp_nm', 'foot_note', 'rcp_dt']  # columns=['crp_cd', 'ind_cd', 'crp_cls', 'crp_nm', 'rpt_nm', 'foot_note', 'rcp_dt']
    qual_ind_var.drop(columns, inplace=True, axis=1)

    # dataset = sparse.load_npz('./merged_FnGuide/qual_matix.npz')

    data_set_file_name = 'quanti_qaul_eps_predict.pkl'
    matched_quanti_data = pd.read_pickle('merged_FnGuide/quanti_eps_predict.pkl')

    matched_quanti_and_qual_data, valid_df_idx_list = pfd.match_fnguide_data_and_delete_redundant(qual_ind_var, matched_quanti_data, data_set_file_name)

    dataset, y = pfd.filter_pos(valid_df_idx_list, matched_quanti_and_qual_data)
    """
    # matched_quanti_and_qual_data = pd.read_pickle('./merged_FnGuide/quanti_qaul_eps_predict.pkl')


    start_time = datetime.now()
    print("start_time : ", start_time)
    rms_list1 = pfd.previous_research_with_svm(matched_quanti_data.values, 30)
    print("take time : {}".format(datetime.now() - start_time))
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
