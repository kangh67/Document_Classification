# === 1. If Arctic sun devices exist in the reviewed reports of 2008-2016 ===

# 9 reports about Arctic appear, 8 were labeled as 0, 1 was labeled as 1


# === 2. What generic names has the product code DWJ in reviewed reports of 2008-2016, and what are there labels

reviwed_reports_0816 = 'data/MAUDE_2008_2016_review.tsv'
MDR_label_0816 = dict()

# get MDR and label
with open(reviwed_reports_0816, 'r') as d:
    line = d.readline()
    while True:
        line = d.readline()
        if not line:
            break
        else:
            thisline = line.split('\t')
            MDR_label_0816[thisline[0]] = thisline[1]

print('=== DWJ product code in reviewed reports 2008-16')
for i in range(2008, 2017):
    with open('MAUDE_data/foidev' + str(i) + '.txt', 'rb') as d:
        d.readline()
        while True:
            line = d.readline().decode('utf8', 'ignore')
            if not line:
                break
            else:
                thisline = line.split('|')
                if '|DWJ|' in line and thisline[0] in MDR_label_0816.keys():
                    print(MDR_label_0816[thisline[0]], line)

