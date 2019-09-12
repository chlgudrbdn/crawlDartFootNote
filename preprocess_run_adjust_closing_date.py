
import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from calendar import monthrange

df3_2 = pd.read_pickle('filter3_2 have_foot_note_case.pkl')
# df4_1 = df3_2[df3_2['dic_cls'] == 'NA']

# df4_2 = pd.DataFrame()

df3_2.sort_values(['crp_cd', 'rpt_nm', 'rcp_dt', 'rcp_no'], ascending=['True', 'True', 'True', 'True'], inplace=True)

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
            tplus_closing_date = datetime(int(t_closing_date[0]), int(t_closing_date[1]), monthrange(int(t_closing_date[0]), int(t_closing_date[1]))[1]) + relativedelta(months=3)

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

