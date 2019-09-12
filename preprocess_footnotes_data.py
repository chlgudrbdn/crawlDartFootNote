import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from calendar import monthrange
from dateutil.relativedelta import relativedelta
import os
import math
from konlpy.tag import Kkma
from konlpy.tag import Komoran
from konlpy.tag import Okt
from konlpy.tag import Hannanum
import re
import gc
# https://replet.tistory.com/70?category=667742 에서 사용해서 import하는 패키지
import statsmodels.api as sm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from math import sqrt
from sklearn.model_selection import KFold
from scipy import sparse
from scipy.sparse import csr_matrix, hstack
from scipy import stats
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
# from thundersvm import *
from sklearn.svm import SVR

# df.loc[i - 1, 'crp_cls'] = crp_cls  # 법인유형(유가증권Y, 코스닥K)
# df.loc[i - 1, 'crp_nm'] = crp_nm  # 공시대상회사(종목명)
# df.loc[i - 1, 'crp_cd'] = crp_cd  # 종목코드
# df.loc[i - 1, 'rpt_nm'] = rpt_nm  # 보고서명
# df.loc[i - 1, 'rcp_no'] = rcp_no  # 보고서 번호. 그러나 첨부파일 번호가 항상 해당 rcp_no에 종속되지 않음.
# df.loc[i - 1, 'dic_cls'] = dic_cls  # 공시구분
# df.loc[i - 1, 'dcm_no'] = dcm_no  # 만약 주석이 다른 첨부파일에 있다면 그 첨부파일의 번호. 추후 재확인을 위함.
# df.loc[i - 1, 'col_dcm_no'] = col_dcm_no  # 만약 연결 주석이 다른 첨부파일에 있다면 그 첨부파일의 번호. 추후 재확인을 위함.
# df.loc[i - 1, 'foot_note'] = foot_note  # 주석
# df.loc[i - 1, 'consolidated_foot_note'] = consolidated_foot_note  # 연결재무제표 주석. 주석과 연결재무제표주석이 있다면 일단 연결재무제표 주석을 우선시해서 크롤링.
# df.loc[i - 1, 'acc_crp'] = acc_crp  # 감사보고서를 작성한 회사.
# df.loc[i - 1, 'rcp_dt']


def filtering_from2013to2018(df):  # 2013~2018만 필터링. 설령 13년에 올라온 공시라도 12년 사업에 대한 것일 가능성이 있음
    # df = pd.read_pickle('total crawlDartFootNote data collection.pkl')
    df1_1 = df[df['rpt_nm'].str.contains('2013') | df['rpt_nm'].str.contains('2014') |
            df['rpt_nm'].str.contains('2015') | df['rpt_nm'].str.contains('2016') |
            df['rpt_nm'].str.contains('2017') | df['rpt_nm'].str.contains('2018')]
    # df3[(int(df3['rpt_nm'].split('(')[1].split('.')[0]) > 2012)]
    # for index, row in df.iterrows():
    #     if int(row['rpt_nm'].split('(')[1].split('.')[0]) < 2013:
    # df5 = df5.sort_values(by=['rcp_dt'], ascending=False)
    # save_with_xlsxwriter('filter from2013to2018.xlsx', df5)
    df['crp_cls'] = df.crp_cls.astype("category")


    df1_1.to_pickle('filter1_1 from2013to2018.pkl')
    return df1_1


def filtering_ind_(df1_1):
    # df1_1 = pd.read_pickle('filter1_1 from2013to2018.pkl')
    df1_2 = df1_1['crp_cd'].drop_duplicates().to_frame()
    df1_2.to_excel('crp_cd list.xlsx', index=False)  # 2062  #
    return df1_2


def filtering_financial_industry(df1_1, df1_2):  # 금융 및 보험업 제거
    # df1_1 = pd.read_pickle('filter1_1 from2013to2018.pkl')
    # df1_2 = pd.read_excel('crp_ind_match.xlsx', dtype=str)
    df2 = pd.merge(df1_2, df1_1)
    print(df2[df2.ind_cd.str.contains('^64')].shape)
    print(df2[df2.ind_cd.str.contains('^65')].shape)
    print(df2[df2.ind_cd.str.contains('^66')].shape)
    df2_1 = df2[~(df2.ind_cd.str.contains('^64') | df2.ind_cd.str.contains('^65') | df2.ind_cd.str.contains('^66'))]
    df2_1 = df2_1.drop(columns=['acc_crp'])  # 일단 외부감사는 모두 받는 것도 아니고 코드가 완전한것도 아니므로 일단 acc_crp 제외
    df2_1.to_pickle('filter2_1 no_fiance_K.pkl')
    return df2_1


def filtering_less_text_data(df2_1):
    # df2_1 = pd.read_pickle('F:/최형규/논문에 사용된 데이터 정리/filter2_1 no_fiance_K.pkl')
    df3_1 = df2_1[df2_1['foot_note'] == 'NA']
    # df3_1 = df2_1[df2_1['foot_note'].isna()]
    # df3_1 = df2_1[df2_1['foot_note'].isnan()]
    # df3_1 = df2_1[pd.isnull(df2_1['foot_note'])]
    writer = pd.ExcelWriter('filter3_1 no_foot_note_case.xlsx', engine='xlsxwriter')  # import xlsxwriter 깔아야 사용 가능한 방법
    df3_1.to_excel(writer, 'Sheet1', index=False)
    writer.save()

    df3_2 = df2_1[~(df2_1['foot_note'] == 'NA')]
    df3_2.to_pickle('filter3_2 have_foot_note_case.pkl')

    return df3_1, df3_2


def revision_disclose_use_or_just_not_resvision(df3_2):  # 만약 다음 공시 이전까지 개정된 것이 있다면 이를 필터링. 그 이후의 개정된 것은 일단 삭제.
    # df3_2 = pd.read_pickle('filter3_2 have_foot_note_case.pkl')
    df4_1 = df3_2[df3_2['dic_cls'] == 'NA']

    # df4_2 = pd.DataFrame()
    df3_2.sort_values(['crp_cd', 'rpt_nm', 'rcp_dt'], ascending=['True', 'True', 'True'], inplace=True)
    pre_crp_cd = ""
    pre_rpt_nm = ""
    # pre_rpt_dt = ""
    pre_index = ""
    index_list_for_delete = []
    for index, row in df3_2.iterrows():
        # row = df3_2.loc[44881, :]  # for test
        cur_crp_cd = row['crp_cd']
        cur_rpt_nm = row['rpt_nm']
        cur_rcp_dt = row['rcp_dt']

        if cur_crp_cd == pre_crp_cd and cur_rpt_nm == pre_rpt_nm:
            t_closing_date = cur_rpt_nm[cur_rpt_nm.find("(")+1:cur_rpt_nm.find(")")].split('.')
            tplus_closing_date = datetime(int(t_closing_date[0]), int(t_closing_date[1]), 1)+ relativedelta(months=3)

            cur_rcp_dt_li = cur_rcp_dt.split('.')
            rpt_dt_datetime = datetime(int(cur_rcp_dt_li[0]), int(cur_rcp_dt_li[1]), int(cur_rcp_dt_li[2]))

            if rpt_dt_datetime < tplus_closing_date :  # 일단 데이터 누수 막기 위해 다음 분기 되기 전 데이터 한정
                # 일단 당장 지울 필요는 없지만 일단은 중복이 있다는 얘기. 즉 대체가 필요하다는 소리
                index_list_for_delete.append(pre_index)  # 즉 중복됐으면서 보다 이전날짜에 나온 공시를 지워야함.
            else:  # 만약 다음 분기 이후에 정정되었다면 일단 이건 쓸모가 없단 소리.
                index_list_for_delete.append(index)
        pre_crp_cd = cur_crp_cd
        pre_rpt_nm = cur_rpt_nm
        # pre_rpt_dt = cur_rcp_dt
        pre_index = index
    # 일단 복잡하므로 후순위로 구현.

    df4_2 = df3_2.drop(index_list_for_delete)
    print(df4_2.shape)

    df3_2.sort_values(['crp_cd', 'rpt_nm', 'rcp_dt', 'rcp_no'], ascending=['True', 'True', 'True', 'True'],
                      inplace=True)
    pre_crp_cd = ""
    pre_rpt_nm = ""
    # pre_rpt_dt = ""
    pre_index = ""
    index_list_for_delete = []
    for index, row in df3_2.iterrows():
        # row = df3_2.loc[44881, :]  # for test
        cur_crp_cd = row['crp_cd']
        cur_rpt_nm = row['rpt_nm']
        cur_rcp_dt = row['rcp_dt']
        if row['dic_cls'] != 'NA':
            if cur_crp_cd == pre_crp_cd and cur_rpt_nm == pre_rpt_nm:
                t_closing_date = cur_rpt_nm[cur_rpt_nm.find("(") + 1:cur_rpt_nm.find(")")].split('.')
                tplus_closing_date = datetime(int(t_closing_date[0]), int(t_closing_date[1]),
                                              monthrange(int(t_closing_date[0]), int(t_closing_date[1]))[
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
    # 보통 t-1년 1~12월 이 4월의 사업보고서에 반영,

    df4_1.to_pickle('filter4_1 without_revision.pkl')
    df4_2.to_pickle('filter4_2 with_revision.pkl')

    return df4_1, df4_2


def if_has_colFootnotes_then_replace(df4_1, df4_2):  # 연결재무제표가 있으면 대체하는가 마는가.
    # df4_1 = pd.read_pickle('filter4_1 without_revision.pkl')
    # df4_2 = pd.read_pickle('filter4_2 with_revision.pkl')
    df5_1 = df4_1.drop(columns=['consolidated_foot_note'])
    df5_2 = df4_1
    df5_2['foot_note'] = df4_1[['foot_note', 'consolidated_foot_note']].apply(
        lambda x: x['consolidated_foot_note'] if x['consolidated_foot_note'] != 'NA' else x['foot_note'], axis=1)
    df5_2 = df5_2.drop(columns=['consolidated_foot_note'])

    df5_3 = df4_2.drop(columns=['consolidated_foot_note'])
    df5_4 = df4_2
    df5_4['foot_note'] = df4_2[['foot_note', 'consolidated_foot_note']].apply(
        lambda x: x['consolidated_foot_note'] if x['consolidated_foot_note'] != 'NA' else x['foot_note'], axis=1)

    df5_1.to_pickle('filter5_1 just footnote.pkl')
    df5_2.to_pickle('filter5_2 alternate col to footnotes.pkl')
    df5_3.to_pickle('filter5_3 just footnote.pkl')
    df5_4.to_pickle('filter5_4 alternate col to footnotes.pkl')
    return df5_1, df5_2, df5_3, df5_4


def divide_by_sector(df, filename, directory_name):
    # df = pd.read_pickle('filter5_2 alternate col to footnotes.pkl')
    # filename = 'filter5_2 alternate col to footnotes'
    # directory_name = './divide_by_sector'
    if not os.path.exists(directory_name):  # bitcoin_per_date 폴더에 저장되도록, 폴더가 없으면 만들도록함.
        os.mkdir(directory_name)
    sector_list = pd.read_excel('한국표준산업분류(10차)_표.xlsx', sheet_name='Sheet1')
    sector_list = list(sector_list['sector'])
    sector_detailed = pd.read_excel('한국표준산업분류(10차)_표.xlsx', sheet_name='Sheet2', dtype=object)
    df_sector_list = []
    for sector in sector_list:
        # sector = sector_list[0] # for test
        detailed_sector_num = list(sector_detailed[sector_detailed['sector'] == sector]['range'])
        detailed_sector_num = [str(x) for x in detailed_sector_num]
        detailed_sector_regex = "|^".join(detailed_sector_num)
        detailed_sector_regex = '^'+detailed_sector_regex
        df1 = df[df.ind_cd.str.contains(detailed_sector_regex, regex=True)]
        print(sector, df1.shape)
        # df1.to_pickle(directory_name+"/"+sector+'_'+filename+'.pkl')
        df_sector_list.append(df1)
    return df_sector_list, sector_list


def divide_by_specific_number(df, max_rows):
    list_of_dfs = []
    while len(df) > max_rows:
        top = df[:max_rows]
        # print(top.shape)
        list_of_dfs.append(top)
        df = df[max_rows:]
        # print(df.shape)
    else:
        list_of_dfs.append(df)
    return list_of_dfs


def foot_note_pos_tagging(df6, morpheme_ref, how_to_pos_treat_as_feature):
    # pos_tagged_lists = []
    for index, row in df6.iterrows():
        if index % 100 == 0:
            print(index)
        # print(index, row)  # for test
        row['foot_note'] = re.sub('\n+', "\n", row['foot_note'])
        row['foot_note'] = re.sub('\n ', "\n", row['foot_note'])
        row['foot_note'] = re.sub(' {2,}', "", row['foot_note'])  # 구   분 같이 표에 명시된 경우가 있는데 이런 걸 제거하기 위함.
        # before_text = "   "  # for test
        # after_text = before_text  # for test
        # after_text = re.sub('[/[\{\}\[\]\/?|\)*~`!\-_+<>@\#$%&\\\=\(\'\"]+', '0', before_text)  # for test
        # after_text = re.sub(' {2,}', '0', before_text)  # for test
        sentences = xplit('. ', '? ', '! ', '\n', '.\n')(row['foot_note'])  # 문장별 구분
        morphemes = []
        i = 0
        for sentence in sentences:  # 이 절차를 미리하는 것은 사실 비효율이지만 에러 처리를 위해 임시방편으로 사용. 그리고 이걸 써야 한나눔의 글자수 제한도 피해갈 수 있다.
            # print(sentence)  # for test
            try:
                if how_to_pos_treat_as_feature == 'use_only_morph':
                    morphemes.extend(morpheme_ref.morphs(sentence))
                elif how_to_pos_treat_as_feature == 'attach_tag_to_pos':
                    # print('pos tagging start')  # for test
                    pos_tag_pair_list = morpheme_ref.pos(sentence)
                    # print(pos_tag_pair_list)  # for test
                    tmp = []
                    for word in pos_tag_pair_list:
                        tmp.append(("/".join(word)))
                    # print('tag result: ', tmp)  # for test
                    morphemes.extend(tmp)
                    del tmp
                    del pos_tag_pair_list
                elif how_to_pos_treat_as_feature == 'seperate_tag_and_pos':
                    morphemes.extend(morpheme_ref.pos(sentence))  # 한쌍의 튜플로 이뤄진 리스트를 품사까지 하나의 feature로 취급하기 위한 작업
            except Exception:
                continue
                print('error index', index, ' ', i, 'th sentence', sentence)
            i += 1
        df6.at[index, 'foot_note'] = morphemes
        del morphemes
        del sentences
        # print(df6.at[index, 'foot_note'])
        # df6.loc[index, 'foot_note'] = morphemes  # bug
        # df6.loc[index].at['morpheme'] = morphemes  # bug
        # pos_tagged_lists.append(morphemes)  # bug
        # df6['morpheme'] = pos_tagged_lists  # bug
    # print(df6['foot_note'])
    # print(df6.info())
    print('done')
    return df6


def test_pos_for_check(df6, morpheme, how_to_pos_treat_as_feature):
    start_time = datetime.now()
    print("start_time : ", start_time)
    if morpheme == 'kkma':
        morpheme_ref = Kkma()  # 꼬꼬마  # 오래걸린다. 제외.
    elif morpheme == 'komoran':
        morpheme_ref = Komoran()  # 코모란
    elif morpheme == 'okt':
        morpheme_ref = Okt()  # 오픈 코리아 텍스트
    elif morpheme == 'hannanum':
        morpheme_ref = Hannanum()  # 한나눔  # 띄어쓰기 처리수준이 낮긴한데 이점이 오히려 합성어가 많은 재무 도메인의 경우 의미가 있을 수 있어 보인다.  # 라고 생각했는데 오류 투성이다.
    columns = ['rcp_no', 'dic_cls', 'dcm_no', 'col_dcm_no', 'consolidated_foot_note']
    df6.drop(columns, inplace=True, axis=1)

    directory_name = './divide_by_sector'
    # filename = 'filter6 '+morpheme+'_'+how_to_pos_treat_as_feature+'.pkl'
    df_list = divide_by_specific_number(df6, 10000)
    del df6
    # df_list_sector, sector_list = divide_by_sector(df6, filename, directory_name)
    # for tmp_df, sector in zip(df_list_sector, sector_list):
    print('divide done')
    len_of_df_list = len(df_list)
    for i in range(len_of_df_list):
        tmp_df = df_list[0]  # 앞에서 하나씩 지우는 pop 같은 방식
        tmp_df = foot_note_pos_tagging(tmp_df, morpheme_ref, how_to_pos_treat_as_feature)
        tmp_df.to_pickle(directory_name + '/filter6 '+morpheme+'_'+how_to_pos_treat_as_feature+'_'+str(i)+'.pkl')
        del tmp_df
        del df_list[0]  # 앞에서 하나씩 지우는 pop 같은 방식
        print("take time : {}".format(datetime.now() - start_time))
        i += 1
        gc.collect()
        # 2번것 다 끝나면
        # 파일 불러올땐 morpheme, how_to_pos_treat_as_feature로 걸러서 출력.
    print("end time : {}".format(datetime.now()))


def check_isnan_or_string(item):
    check_nan = True
    try:
        check_nan = math.isnan(item)  # nan이면 True
    except Exception:
        print(item)
    return check_nan


def substitute_main():  # fnguide에서 긁어온 종속 변수 손보기.
    path_dir = 'C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable retrived 2019_05_15'  # done (파일사이즈 문제와 전처리 편의를 위해 pickle로 저장하게 함.)
    file_df_list = []
    file_name_list = []
    for path, dirs, files in os.walk(path_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.xlsx':
                if 'substitute' in file.split(".")[0]:
                    tmp_df = pd.read_excel(path + "\\" + file, dtype=object)  # dtype=str option이 없으면 종목코드가 숫자로 처리됨.
                    if not tmp_df.empty:
                        print(file)
                        file_name_list.append(file)
                        file_df_list.append(tmp_df)
    for tmp_df, file_name in zip(file_df_list, file_name_list):
        # tmp_df = pd.read_excel(path_dir+"\\"+'PER_rawData.xlsx')  # dtype=str option이 없으면 종목코드가 숫자로 처리됨.
        main_dat = tmp_df.loc[:, tmp_df.columns.str.contains('^M')]
        noncon3_dat = tmp_df.loc[:, tmp_df.columns.str.contains('^3')]
        con6_dat = tmp_df.loc[:, tmp_df.columns.str.contains('^6')]
        for index, row in main_dat.iterrows():
            for main in list(main_dat.columns):
                if check_isnan_or_string(row[main]):  # nan인 경우 같은 줄, 같은 item에 연결 또는 단독 쪽 데이터를 체크해서 있으면 끌어온다.
                    item = main.split('-')[1]
                    for noncon3 in list(noncon3_dat.columns):
                        if item == noncon3.split('-')[1] and \
                            not check_isnan_or_string(noncon3_dat.loc[index, noncon3]):  # 혹시라도 같은 항목에 값이 있다면 보삽.
                            row[main] = noncon3_dat.loc[index, noncon3]  # 일단은 개별 재무제표 값이 있으면 이걸로 갈아 끼운다.
                    for con6 in list(con6_dat.columns):
                        if item == con6.split('-')[1] and \
                            not check_isnan_or_string(con6_dat.loc[index, con6]):  # 혹시라도 같은 항목에 값이 있다면 보삽.
                            row[main] = con6_dat.loc[index, con6]  # 일단은 연결 재무제표 값이 있으면 이걸로 갈아 끼운다.
        main_dat['결산월'] = list(tmp_df['결산월'])
        main_dat['회계년'] = list(tmp_df['회계년'])
        main_dat['Symbol'] = list(tmp_df['Symbol'])
        # file_name = 'PER_rawData.xlsx'  # for test
        main_dat.to_excel('./financial ratio for dependent variable retrived 2019_05_15/substitute_'+file_name)  # 보충하긴 했는데 이게 너무 생뚱맞는 값을 뱉는거 같다.
        print('save done')
    # tmp_df.loc[30371].at['M000701020-PER(연율화)(배)']
    # tmp_df.loc[30372].at['M000701020-PER(연율화)(배)']


def statical_analysis():
    matched_quanti_data = pd.read_pickle('merged_FnGuide/quanti_eps_predict.pkl')
    X = matched_quanti_data[['M000909001-영업활동으로인한현금흐름(천원)']]
    # X = matched_quanti_data[['M000909001-영업활동으로인한현금흐름(천원)', 'M000909018-재무활동으로인한현금흐름(천원)', 'M000909015-투자활동으로인한현금흐름(천원)']]
    y = matched_quanti_data['M000601003-수정EPS(원)']
    X = sm.add_constant(X)
    result = sm.OLS(np.asarray(y), X).fit()
    print(result.rsquared)
    # 요약결과 출력
    print(result.summary())


def match_fnguide_data_and_delete_redundant(df, quanti_data, file_name):  # t+1 종속변수와 t독립 변수 비교
    result_df = pd.DataFrame()
    valid_df_idx_list = []
    for index, row in df.iterrows():
        rpt_nm = row['rpt_nm']
        t_closing_date = rpt_nm[rpt_nm.find("(")+1:rpt_nm.find(")")].split('.')
        t_closing_date = datetime(int(t_closing_date[0]), int(t_closing_date[1]), 1)  # t+1이 아니다. 오는 값은 꽉차있고 이미 매칭된 정량+종속 변수 값이다. 정량변수의 식별 정보와 맞춰주면 된다.
        tplus_data = quanti_data[(quanti_data['Symbol'] == str(row['crp_cd'])) &
                                 (quanti_data['결산월'] == t_closing_date.month) &
                                 (quanti_data['회계년'] == t_closing_date.year)]
        # tplus_data.drop(['rpt_nm', 'Symbol', '결산월', '회계년'], inplace=True, axis=1)
        if tplus_data.shape[0] > 1:
            print(tplus_data)
        if tplus_data.empty:
            result_df = result_df.append(pd.Series(), ignore_index=True)
            continue
        valid_df_idx_list.append(index)  # 최종적로는 이것만 있으면 된다. # 일단 적절한 값이 없는 경우 알아서 생략되도록 앞의 코드에서 처리.
        result_df = result_df.append(tplus_data, ignore_index=True)
    df.reset_index(drop=True, inplace=True)
    result_df.reset_index(drop=True, inplace=True)
    df = pd.concat([df, result_df], axis=1)
    columns = ['crp_cd', 'rpt_nm', 'Symbol', '결산월', '회계년']
    # columns = ['crp_cd', 'ind_cd', 'crp_cls', 'crp_nm', 'rpt_nm', 'rcp_no', 'dic_cls', 'dcm_no', 'col_dcm_no', 'consolidated_foot_note', 'rcp_dt',
    #            'Symbol', '결산월', '회계년']
    # df5_2의 컬럼 목록 ['crp_cd', 'ind_cd', 'crp_cls', 'crp_nm', 'rpt_nm', 'rcp_no', 'dic_cls', 'dcm_no', 'col_dcm_no', 'foot_note', 'consolidated_foot_note', 'rcp_dt']
    # 현시점에서 산업코드 나누는건 의미 없고(사실 절반이 제조업이라 더더욱), 종목코드는 중복이라 Symbol 삭제,
    # rcp_dt는 이전 단계에 써먹어야 했음, crp_nm은 어차피 종목코드로 대체(검색 편의를 위해 남겼을 뿐),
    # 결산월과 회계년은 이미 t+1과 t0를 맞추는데 사용.
    # rpt_nm은 좀 애매한데 일단 분기 보고서인지 반기보고서인지 나눠서 제어할 필요가 있다고 보고 남김.
    df.drop(columns, inplace=True, axis=1)
    df.dropna(inplace=True)  # 사실 별 의미 없는 짓이다.
    directory_name = './merged_FnGuide'
    if not os.path.exists(directory_name):  # bitcoin_per_date 폴더에 저장되도록, 폴더가 없으면 만들도록함.
        os.mkdir(directory_name)
    df.to_pickle(directory_name+'/'+file_name)
    print(len(valid_df_idx_list))
    print(df.shape)
    # np.save('./merged_FnGuide/'+file_name, df.values)
    # for test
    # directory_name = 'merged_FnGuide'
    # file_name = 'filter6 komoran_attach_tag_to_pos_0.pkl'
    # df = pd.read_pickle(directory_name+'/merged_FnGuide '+file_name)
    # print(df.shape)
    # main_dat = df.loc[:, df.columns.str.contains('^M')]
    # df = df.dropna(thresh=len(df.columns) - len(main_dat.columns)+1)  # 종속변수가 모두 nan이려면 non-NaN인 값이 5보다 적은 경우이다.
    # print(main_dat.dropna(how='all').shape)
    # print(df.shape)
    # df.to_pickle(directory_name+'/merged_FnGuide '+file_name)
    # print(df1.loc[0])
    # for test
    return df, valid_df_idx_list


def match_fnguide_data_among_them(quanti_ind_var, quanti_dep_var, keyword, file_name):
    # df1 = pd.read_excel('C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\previous research independant variable\\about_EPS_independant_var.xlsx', dtype=object, sheet_name='Sheet1')
    # df2 = pd.read_excel('C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\financial ratio for dependent variable retrived 2019_05_15\\EPS_rawData.xlsx', dtype=object, sheet_name='Sheet1')
    # keyword = '수정EPS\(원\)'
    # print(quanti_ind_var.columns)
    # print(quanti_dep_var.columns)
    quanti_dep_var = quanti_dep_var.drop(columns=list(quanti_dep_var.loc[:, quanti_dep_var.columns.str.contains('^주기')].columns))  # 주기는 불필요하니 제거
    quanti_dep_var = pd.concat([quanti_dep_var.loc[:, ['Symbol', 'Name', '결산월', '회계년']], quanti_dep_var.loc[:, quanti_dep_var.columns.str.contains(keyword)]], axis=1)
    quanti_dep_var = quanti_dep_var.replace(0, np.nan)  # 가끔 데이터가 없는데 0으로 채워진 경우는 그냥 이렇게 한다. 어차피 데이터가 없을게 뻔해서 의미는 없지만 혹시 모르니까.
    quanti_dep_var.dropna(thresh=5, inplace=True)  # 식별 위한 정보를 제외한 것이 없는 경우. 어차피 이게 없으면 아무것도 안되므로.

    # print(quanti_dep_var.columns)

    quanti_ind_var = quanti_ind_var.drop(columns=list(quanti_ind_var.loc[:, quanti_ind_var.columns.str.contains('^주기')].columns))  # 주기는 불필요하니 제거
    quanti_ind_var = quanti_ind_var.replace(0, np.nan)  # 가끔 데이터가 없는데 0으로 채워진 경우는 그냥 이렇게 한다. 어차피 데이터가 없을게 뻔해서 의미는 없지만 혹시 모르니까.
    quanti_ind_var.dropna(thresh=5, inplace=True)  # 식별 위한 정보를 제외한 것이 없는 경우. 어차피 이게 없으면 아무것도 안되므로.
    # print(quanti_ind_var.columns)

    result_df = pd.DataFrame()
    for index, row in quanti_ind_var.iterrows():
        tplus_closing_date = datetime(int(row['회계년']), int(row['결산월']), 1) + relativedelta(months=3)
        tplus_data = quanti_dep_var[(quanti_dep_var['Symbol'] == row['Symbol']) &
                                    (quanti_dep_var['결산월'] == tplus_closing_date.month) &
                                    (quanti_dep_var['회계년'] == tplus_closing_date.year)]
        if tplus_data.shape[0] > 1:  # 중복된 월, 일의 데이터 없는 문제 확인.
            print(tplus_data)  # 일단 미리 없애놨으니 나타날린 없지만 그래도 한다.
        if tplus_data.empty:
            result_df = result_df.append(pd.Series(), ignore_index=True)  # 매칭하는 날짜는 있는데 비었다면 빈칸으로 채우고 넘어간다.
            continue
        result_df = result_df.append(tplus_data, ignore_index=True)  # 찾은 결과를 한줄씩 붙인뒤 나중에 옆으로 붙일 예정.
    result_df = result_df.drop(columns=['Symbol', 'Name', '결산월', '회계년'])  # 종속변수 쪽 식별 정보는 필요 없음.
    result_df.reset_index(drop=True, inplace=True)
    quanti_ind_var.reset_index(drop=True, inplace=True)
    quanti_ind_var = quanti_ind_var.drop(columns=['Name'])  # 그냥 종목번호보다 보기좋아서 냅둔거라. 지워도 상관없음.

    quanti_ind_var = pd.concat([quanti_ind_var, result_df], axis=1)
    quanti_ind_var.dropna(inplace=True)  # 이전 단계에서 걸렀을 가능성이 높지만 그래도 1~2개 없는 경우를 거르기 위함.

    # directory_name = 'merged_FnGuide ind_var/'
    # if not os.path.exists(directory_name):
    #     os.mkdir(directory_name)
    # df1.to_excel(directory_name+'/merged_FnGuide ind_var '+keyword+'.xlsx')
    directory_name = './merged_FnGuide'
    if not os.path.exists(directory_name):  # bitcoin_per_date 폴더에 저장되도록, 폴더가 없으면 만들도록함.
        os.mkdir(directory_name)
    # np.save('./merged_FnGuide/'+file_name, quanti_ind_var.values)
    quanti_ind_var.to_pickle(directory_name+'/'+file_name)

    print(quanti_ind_var.columns)
    return quanti_ind_var  # 다음에 쓰려고 ndarray로 반환하지 않음.


### evaluation ###
# revenueMatrix = np.array(RevenueDistributionPerWorld)
# 참고 http://www.dodomira.com/2016/04/02/r%EC%9D%84-%EC%82%AC%EC%9A%A9%ED%95%9C-t-test/
def equ_var_test_and_unpaired_t_test(x1, x2):  # 모든 조합으로 독립표본 t-test 실시. 일단 다른 변수로 감안.(같다면 등분산 t-test라고 생각)
    # 등분산성 확인. 가장 기본적인 방법은 F분포를 사용하는 것이지만 실무에서는 이보다 더 성능이 좋은 bartlett, fligner, levene 방법을 주로 사용.
    # https://datascienceschool.net/view-notebook/14bde0cc05514b2cae2088805ef9ed52/
    if stats.bartlett(x1, x2).pvalue < 0.05:
        tTestResult = stats.ttest_ind(x1, x2, equal_var=True)
        print("The t-statistic and p-value assuming equal variances is %.3f and %.3f." % tTestResult)
        # 출처: http: // thenotes.tistory.com / entry / Ttest - in -python[NOTES]
        if tTestResult.pvalue < 0.05:
            compare_mean(x1, x2)
        else:
            print("two sample mean is same h0 not rejected")
    else:
        tTestResult = stats.ttest_ind(x1, x2, equal_var=False)  # 등분산이 아니므로 Welch’s t-test
        print("The t-statistic and p-value not assuming equal variances is %.3f and %.3f" % tTestResult)
        # 출처: http: // thenotes.tistory.com / entry / Ttest - in -python[NOTES]
        if tTestResult.pvalue < 0.05:
            compare_mean(x1, x2)
        else:
            print("two sample mean is same h0 not rejected")
# from sklearn.preprocessing import MinMaxScaler
#
# dataset = scaler.fit_transform(dataset)
# trainPredict = scaler.inverse_transform(trainPredict)
def nested_cv(X, y, inner_cv, outer_cv, parameter_grid):
    outer_scores = []

    # y = y.reshape(-1, 1)
    # outer_cv의 분할을 순회하는 for 루프
    # (split 메소드는 훈련과 테스트 세트에 해당하는 인덱스를 리턴합니다)
    for training_samples, test_samples in outer_cv.split(X, y):
        # 최적의 매개변수를 찾습니다
        best_parms = {}
        best_score = -np.inf
        # 매개변수 그리드를 순회합니다
        # GridSearchCV(SVR)  # 병렬화. 일단 후순위.
        for parameters in parameter_grid:
            # 안쪽 교차 검증의 점수를 기록합니다
            cv_scores = []
            # inner_cv의 분할을 순회하는 for 루프
            for inner_train, inner_test in inner_cv.split(
                    X[training_samples], y[training_samples]):
                # 훈련 데이터와 주어진 매개변수로 분류기를 만듭니다
                reg = SVR(**parameters)
                reg.fit(X[inner_train], y[inner_train])
                # 검증 세트로 평가합니다
                score = reg.score(X[inner_test], y[inner_test])  # SVR의 기본 성능 평가 척도는 R^2이다. 보통은 0~1(성능이 너무 구리면 마이너스로도 간다)
                cv_scores.append(score)
            # 안쪽 교차 검증의 평균 점수를 계산합니다
            mean_score = np.mean(cv_scores)
            if mean_score > best_score:
                # 점수가 더 높은면 매개변수와 함께 기록합니다
                best_score = mean_score
                best_params = parameters
        # 바깥쪽 훈련 데이터 전체를 사용해 분류기를 만듭니다
        reg = SVR(**best_params)
        print(best_params)
        reg.fit(X[training_samples], y[training_samples])
        # 테스트 세트를 사용해 평가합니다
        # outer_scores.append(reg.score(X[test_samples], y[test_samples]))
        outer_scores.append(sqrt(mean_squared_error(reg.predict(X[test_samples]), y[test_samples])))
        #, sqrt(mean_squared_error(reg.predict(X[test_samples]), y[test_samples]))))  # 이중 리스트로, 첫번째는 r^2 두번째는 rmse
    print('outer_scores: ', outer_scores)
    return np.mean(outer_scores)  # 전체 데이터 셋 대상으로 한 test의 예측값.


def compare_mean(x1, x2):
    mean1 = np.mean(x1)
    mean2 = np.mean(x2)
    print("mean of first  one is ", mean1)
    print("mean of second one is ", mean2)
    if mean1 < mean2:
        print("so mean2 is bigger")
    else:
        print("so mean1 is bigger")


def previous_research_with_svm(dataset, try_cnt):
    # try_cnt = 30  # for test
    # param_grid = [{'kernel': ['rbf'],  # rbf면 c와 gamma 쓴다.
    #                'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #                'gamma': ['auto']},
    #               {'kernel': ['linear'],  # c만 사용.
    #                'C': [0.001, 0.01, 0.1, 1, 10]},
    #               {'kernel': ['poly'],
    #                'C': [0.001, 0.01, 0.1, 1, 10],
    #                'gamma': ['auto'],
    #                'degree': [2, 3, 4],
    #                'epsilon': [0.01, 0.1, 0.2, 0.5, 1]}
    #               ]
    param_grid = [{'kernel': ['rbf'],
                   'gamma': ['auto']}
                  ]

    over_random_state_try = []
    scaler = StandardScaler()
    for seed in range(try_cnt):
        # seed = 42  # for test
        kf = KFold(n_splits=5, random_state=seed, shuffle=True)
        average_kfold_train_test_score_with_highest_hyperparam_of_train_val = \
            nested_cv_multiprocess(scaler.fit_transform(dataset[:, 3:-1]), dataset[:, -1].ravel(), kf, kf, param_grid)
            # nested_cv(scaler.fit_transform(dataset[:, 3:-1]), dataset[:, -1].ravel(), kf, kf, ParameterGrid(param_grid))
        # X_train, X_test, y_train, y_test = \
        #     train_test_split(df.iloc[:, 4:-1], df.iloc[:, -1], test_size=0.2, random_state=seed)  # 제대로 처리됐다면 ['Symbol', 'Name', '결산월', '회계년'] 순이 될 것.
        # best_score = 0
        over_random_state_try.append(average_kfold_train_test_score_with_highest_hyperparam_of_train_val)
        # print(rms)
        # if score > best_score:
        #     best_score = score
        #     best_parameters = {'C':C, 'gamma': gamma}
        # print("최고 점수: {:.2f}".format(best_score))
        # print("최적 매개점수: {}".format(best_parameters))
    # 교차검증 테스트 평균 점수가 같은데, 표준편차가 작다면 그게 더 좋다고 판단.
    # 탐색순서는 C가 고정되고 gamma가 변하는 식.
    return over_random_state_try


def xplit(*delimiters):
    return lambda value: re.split('|'.join([re.escape(delimiter) for delimiter in delimiters]), value)


def identity_tokenizer_with_komoran(sentences):
    morpheme_ref = Komoran()  # 코모란
    sentences = xplit('. ', '? ', '! ', '\n', '.\n')(sentences)  # 문장별 구분
    morphemes = []
    # i = 0
    for sentence in sentences:  # 이 절차를 미리하는 것은 사실 비효율이지만 에러 처리를 위해 임시방편으로 사용. 그리고 이걸 써야 한나눔의 글자수 제한도 피해갈 수 있다.
        # print(sentence)  # for test
        try:
            morphemes.extend(morpheme_ref.nouns(sentence))
            """
            if how_to_pos_treat_as_feature == 'use_only_morph':
                morphemes.append(morpheme_ref.morphs(sentence))
            elif how_to_pos_treat_as_feature == 'attach_tag_to_pos':
                # print('pos tagging start')  # for test
                # pos_tag_pair_list = morpheme_ref.pos(sentence)
                pos_tag_pair_list = morpheme_ref.nouns(sentence)
                # print(pos_tag_pair_list)  # for test
                tmp = []
                for word in pos_tag_pair_list:
                    tmp.append(("/".join(word)))
                # print('tag result: ', tmp)  # for test
                morphemes.append(tmp)
                del tmp
                del pos_tag_pair_list
            elif how_to_pos_treat_as_feature == 'seperate_tag_and_pos':
                morphemes.append(morpheme_ref.pos(sentence))  # 한쌍의 튜플로 이뤄진 리스트를 품사까지 하나의 feature로 취급하기 위한 작업
            """
        except Exception:
            continue
            # print('error index', index, ' ', i, 'th sentence', sentence)
        # i += 1
    return morphemes


def nested_cv_multiprocess(X, y, inner_cv, outer_cv, parameter_grid, seed):
    outer_scores = []
    print(X.shape)
    # outer_cv의 분할을 순회하는 for 루프
    # (split 메소드는 훈련과 테스트 세트에 해당하는 인덱스를 리턴합니다)
    # X = X.toarray()  # 늦어질 뿐이다
    """
    # 정량적인 데이터만 쓸 경우
    for training_samples, test_samples in outer_cv.split(X, y):
        # 최적의 매개변수를 찾습니다
        grid_search = GridSearchCV(SVR(), parameter_grid, cv=inner_cv, n_jobs=-1)
        grid_search.fit(X[training_samples], y[training_samples])

        # 바깥쪽 훈련 데이터 전체를 사용해 분류기를 만듭니다
        print("최적 매개변수:", grid_search.best_params_)
        print("최고 교차 검증 점수: {:.5f}".format(grid_search.best_score_))
        reg = SVR(**grid_search.best_params_)
        reg.fit(X[training_samples], y[training_samples])
        # 테스트 세트를 사용해 평가합니다
        # outer_scores.append(reg.score(X[test_samples], y[test_samples]))
        print('R^2 of this param : ', reg.score(X[test_samples], y[test_samples]))
        outer_scores.append(sqrt(mean_squared_error(reg.predict(X[test_samples]), y[test_samples])))
        #, sqrt(mean_squared_error(reg.predict(X[test_samples]), y[test_samples]))))  # 이중 리스트로, 첫번째는 r^2 두번째는 rmse
    """

    # 정성적인데이터도 쓸 경우
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed)
    print('split done')
    # score = cross_val_score(SVR(kernel='rbf', gamma='auto'), X_train, y_train, cv=5)
    # print("최고 교차 검증 점수: ", score)
    svr = SVR(kernel='rbf', gamma='auto', C=1.0, epsilon=0.1)
    svr.fit(X_train, y_train)
    print('R^2 score : ', svr.score(X_test, y_test))
    outer_scores.append(sqrt(mean_squared_error(svr.predict(X_test), y_test)))
    print('outer_scores: ', outer_scores)


    '''

    for training_samples, test_samples in outer_cv.split(X, y):
        # 최적의 매개변수를 찾습니다
        # grid_search = GridSearchCV(SVR(), parameter_grid, cv=inner_cv, n_jobs=-1)
        # print(training_samples.tolist())
        # scores = cross_val_score(SVR(**parameter_grid), X.todok()[training_samples.tolist()], y.todok()[training_samples.tolist()], cv=inner_cv)
        # grid_search.fit(X.todok()[training_samples.tolist()], y.todok()[training_samples.tolist()])
        # 바깥쪽 훈련 데이터 전체를 사용해 분류기를 만듭니다
        # print("교차 검증 점수: ", scores)
        reg = SVR(kernel='rbf', gamma='auto')
        reg.fit(X.todok()[training_samples.tolist()], y.todok()[training_samples.tolist()])
        # 테스트 세트를 사용해 평가합니다
        # outer_scores.append(reg.score(X[test_samples], y[test_samples]))
        print('R^2 of this param : ', reg.score(X.todok()[training_samples.tolist()], y.todok()[training_samples.tolist()]))
        # 테스트 세트를 사용해 평가합니다
        outer_scores.append(sqrt(mean_squared_error(reg.predict(X.todok()[test_samples.tolist()]), y.todok()[test_samples.tolist()])))
    print('outer_scores: ', outer_scores)
    '''

    return np.mean(outer_scores)  # 전체 데이터 셋 대상으로 한 test의 예측값.


def svm_with_foot_note(X, y, try_cnt):  # https://data-newbie.tistory.com/32 이쪽도 참고바람.
    # param_grid = [{'kernel': ['rbf'],  # rbf면 c와 gamma 쓴다.
    #                'C': [0.001, 0.01, 0.1, 1, 10, 100],
    #                'gamma': [0.001, 0.01, 0.1, 1, 10, 'scale']},
    #               {'kernel': ['linear'],  # c만 사용.
    #                'C': [0.001, 0.01, 0.1, 1, 10]},
    #               {'kernel': ['poly'],
    #                'C': [0.001, 0.01, 0.1, 1, 10],
    #                'gamma': [0.001, 0.01, 0.1, 1, 10, 'scale'],
    #                'degree': [2, 3, 4],
    #                'epsilon': [0.01, 0.1, 0.2, 0.5, 1]}
    #               ]
    param_grid = [{'kernel': ['rbf'],
                   'gamma': ['auto']}]
    over_random_state_try = []
    # tfidf = TfidfVectorizer(tokenizer=identity_tokenizer_with_komoran, lowercase=False)
    # dataset = sparse.csr_matrix(tfidf.fit_transform(df['foot_note'].values))
    for seed in range(try_cnt):
        kf = KFold(n_splits=5, random_state=seed, shuffle=True)
        average_kfold_train_test_score_with_highest_hyperparam_of_train_val = \
            nested_cv_multiprocess(X, y, kf, kf, param_grid, seed)
        over_random_state_try.append(average_kfold_train_test_score_with_highest_hyperparam_of_train_val)
    return over_random_state_try


def identity_tokenizer(text):
    return text


def filter_nouns_already_tagged(valid_df_idx_list, matched_quanti_and_qual_data):
    '''
    file_df_list = []
    file_name_list = []
    for path, dirs, files in os.walk(path_dir):
        for file in files:
            if os.path.splitext(file)[1].lower() == '.pkl':
                if 'komoran' in file.split(".")[0]:
                    tmp_df = pd.read_pickle(path + "\\" + file)  # dtype=str option이 없으면 종목코드가 숫자로 처리됨.
                    if not tmp_df.empty:
                        print(file)
                        file_name_list.append(file)
                        file_df_list.append(tmp_df)
                        del tmp_df
                        gc.collect()
    file_df_list = pd.concat(file_df_list, join='inner', axis=0)
    file_df_list.to_pickle(path_dir + '\\filter6 komoran_attach_tag_to_pos.pkl')
    del file_df_list
    '''
    '''
    '''
    path_dir = 'C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\divide_by_sector'  # done (파일사이즈 문제와 전처리 편의를 위해 pickle로 저장하게 함.)
    df = pd.read_pickle(path_dir + '\\filter6 komoran_attach_tag_to_pos.pkl')
    # df = pd.read_pickle('./merged_FnGuide/quanti_qaul_eps_predict.pkl')
    df = df.iloc[valid_df_idx_list]
    tfidf = TfidfVectorizer(tokenizer=identity_tokenizer, lowercase=False, max_df=0.95, min_df=0)
    tdm = sparse.csr_matrix(tfidf.fit_transform(list(df['foot_note'])))
    sparse.save_npz('./merged_FnGuide/qual_matix.npz', tdm)

    tdm = sparse.load_npz('./merged_FnGuide/qual_matix.npz')
    scaler = StandardScaler()
    quanti_scaled = scaler.fit_transform(matched_quanti_and_qual_data.values[:, 0:-1])
    # quanti_scaled = np.insert(quanti_scaled, quanti_scaled.shape[1], list(df.values[:, -1]), axis=1)  # 종속변수는 그냥 따로 리턴하는게 속편하다.
    quanti_sparse_matrix = csr_matrix(quanti_scaled.tolist())
    X = hstack([tdm, quanti_sparse_matrix])
    sparse.save_npz('./merged_FnGuide/dataset.npz', X)
    '''
    path_dir = 'C:\\Users\\jin\\PycharmProjects\\crawlDartFootNote\\divide_by_sector\\'
    file_name = 'filter6 komoran_attach_tag_to_pos_6.pkl'
    df = pd.read_pickle(path_dir + file_name)
    # df.reset_index(inplace=True, drop=True)
    for index, row in df.iterrows():
        matching = [s for s in row['foot_note'] if ("/N") in s]
        # print(matching)
        df.loc[index, 'foot_note'] = matching
    df.to_pickle(path_dir + file_name)
    '''
    y = matched_quanti_and_qual_data.values[:, -1]
    return X, y  # 종속변수와 TDM과 quanti 데이터 합친 sparse matrix. 전자를 y, 후자를 X취급하면 그대로 학습이 가능.
