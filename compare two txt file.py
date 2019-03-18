i=0
cnt=0
# f1 = open("2018 03 30 사업보고서의 감사보고서의 주석.txt", 'r')
# f2 = open("2018 03 30 사업보고서의 주석.txt", 'r')
f1 = open("2018 03 30 사업보고서의 감사보고서의 연결재무제표 주석.txt", 'r')
f2 = open("2018 03 30 사업보고서의 연결제무제표 주석.txt", 'r')
while True:
    line1 = f1.readline()
    line2 = f2.readline()
    i += 1
    if line1 != line2:

        print('line', i)
        print('line1', line1)
        print('line2', line2)
        cnt += 1
    if not line1 or not line2:
        print(cnt)
        break

# 1. 감사보고서의 주석과 사업보고서의 주석은 동일한 것으로 판단됨.
"""
from scipy import spatial

f1 = open("2018 03 30 사업보고서의 감사보고서의 주석.txt", 'r')
f2 = open("2018 03 30 사업보고서의 주석.txt", 'r')

dataSetI = f1.readlines()
dataSetII = f2.readlines()
result = 1 - spatial.distance.cosine(dataSetI, dataSetII)
"""
