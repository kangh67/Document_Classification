import re


# ========== Prepare inclusion and exclusion keywords, filter function
# Inclusion and exclusion keywords
in_gen = 'data/inclusion_generic.txt'
ex_gen = 'data/exclusion_generic.txt'
in_manu = 'data/inclusion_manu.txt'

# dict of inclusion and exclusion keywords
inclusion_generic = dict()
exclusion_generic = dict()
inclusion_manu = dict()

# inclusion_generic
with open(in_gen, 'r') as d:
    while True:
        line = d.readline()
        if not line:
            break
        else:
            inclusion_generic[line.replace('\n', '').lower()] = 0
# exclusion_generic
with open(ex_gen, 'r') as d:
    while True:
        line = d.readline()
        if not line:
            break
        else:
            exclusion_generic[line.replace('\n', '').lower()] = 0
# inclusion_manu
with open(in_manu, 'r') as d:
    while True:
        line = d.readline()
        if not line:
            break
        else:
            inclusion_manu[line.replace('\n', '').lower()] = 0

# Summary of inclusion and exclusion keywords
print('=== Summary of inclusion and exclusion keywords ===')
print('Inclusion keywords - generic names:', len(inclusion_generic))
print('Exclusion keywords - generic names:', len(exclusion_generic))
print('Inclusion keywords - manufacturer names:', len(inclusion_manu))

# Count generic and manu hit of filtered data
generic_inclu_hit = set()
manu_inclu_hit = set()


# Function of filter application
def apply_filter(line):
    brand = line.split('|')[6].lower()
    generic = line.split('|')[7].lower()
    manu = line.split('|')[8].lower()
    flag_in_ge = False
    flag_in_manu = False
    flag_ex = False

    # exclusion
    for keyword in exclusion_generic.keys():
        if re.search(r'\b' + keyword + r'\b', brand) or re.search(r'\b' + keyword + r'\b', generic):
            exclusion_generic[keyword] += 1
            flag_ex = True
    if flag_ex:
        return False

    # inclusion
    for keyword in inclusion_generic.keys():
        if re.search(r'\b' + keyword + r'\b', brand) or re.search(r'\b' + keyword + r'\b', generic):
            inclusion_generic[keyword] += 1
            flag_in_ge = True
            # Count generic inclusion keyword hit
            generic_inclu_hit.add(keyword)
    for keyword in inclusion_manu.keys():
        if re.search(r'\b' + keyword + r'\b', manu):
            inclusion_manu[keyword] += 1
            flag_in_manu = True
            # Count manu inclusion keyword hit
            manu_inclu_hit.add(keyword)

    return flag_in_ge or flag_in_manu


# ========== Interpret MAUDE data year by year
# MAUDE original data
dev_root = 'MAUDE_data/foidev'
text_root = 'MAUDE_data/foitext'

mdr_all = dict()
generic_all = dict()
brand_all = dict()
manu_all = dict()


def interpret_dev(year):
    mdr_current = dict()
    generic_current = dict()
    generic_new_current = dict()
    brand_current = dict()
    brand_new_current = dict()
    manu_current = dict()
    manu_new_current = dict()

    generic_inclu_hit.clear()
    manu_inclu_hit.clear()

    filtered_data = []
    filtered_mdr = set()
    filtered_gen = set()
    filtered_gen_new = set()
    filtered_brand = set()
    filtered_brand_new = set()
    filtered_manu = set()
    filtered_manu_new = set()

    with open(dev_root + year + '.txt', 'rb') as d:
        d.readline()
        while True:
            line = d.readline().decode('utf8', 'ignore')
            if not line:
                break
            else:
                l = line.split('|')
                if len(l) < 9:
                    continue
                mdr_all.setdefault(l[0], 0)  # mdr
                mdr_current.setdefault(l[0], 0)
                generic_all.setdefault(l[7], 0)  # generic name
                generic_current.setdefault(l[7], 0)
                brand_all.setdefault(l[6], 0)  # brand name
                brand_current.setdefault(l[6], 0)
                manu_all.setdefault(l[8], 0)  # manufacturer name
                manu_current.setdefault(l[8], 0)
                mdr_all[l[0]] += 1
                mdr_current[l[0]] += 1
                generic_all[l[7]] += 1
                generic_current[l[7]] += 1
                brand_all[l[6]] += 1
                brand_current[l[6]] += 1
                manu_all[l[8]] += 1
                manu_current[l[8]] += 1

                if generic_all[l[7]] == 1:
                    generic_new_current[l[7]] = 1
                if brand_all[l[6]] == 1:
                    brand_new_current[l[6]] = 1
                if manu_all[l[8]] == 1:
                    manu_new_current[l[8]] = 1

                if apply_filter(line) and l[0] not in filtered_mdr:
                    filtered_mdr.add(l[0])
                    filtered_gen.add(l[7])
                    if l[7] in generic_new_current.keys():
                        filtered_gen_new.add(l[7])
                    filtered_brand.add(l[6])
                    if l[6] in brand_new_current.keys():
                        filtered_brand_new.add(l[6])
                    filtered_manu.add(l[8])
                    if l[8] in manu_new_current.keys():
                        filtered_manu_new.add(l[8])
                    filtered_data_one = [year, l[0], l[7], l[6], l[8]]
                    filtered_data.append(filtered_data_one)
        summary = [year, len(mdr_current), len(filtered_data), len(generic_current), len(filtered_gen),
                   len(generic_new_current), len(filtered_gen_new), len(brand_current), len(filtered_brand),
                   len(brand_new_current), len(filtered_brand_new), len(manu_current), len(filtered_manu),
                   len(manu_new_current), len(filtered_manu_new), len(generic_inclu_hit), len(manu_inclu_hit)]

    # filtered_data[mdr# in a year][0:year, 1:mdr, 2:generic, 3:brand, 4:manu]
    # summary[0:year, 1:mdr#, 2:filtered_mdr#, 3:gen#, 4:filtered_gen#, 5:new_gen#, 6:filtered_new_gen#, 7:brand#,
    #           8:filtered_brand#, 9:new_brand#, 10:filtered_new_brand#, 11:manu#, 12:filtered_manu#,
    #           13:new_manu#, 14:filtered_new_manu#, 15:activated_gene_key#, 16:activated_manu_key#]
    return filtered_data, summary


def interpret_text(filtered_data, year):
    text_data = dict()
    # get all mdr
    for entry in filtered_data:
        text_data[entry[1]] = 'N/A'
    # match text for each mdr
    with open(text_root + year + '.txt', 'rb') as d:
        d.readline()
        while True:
            line = d.readline().decode('utf8', 'ignore')
            if not line:
                break
            else:
                line = line.split('|')
                if line[0] in text_data.keys():
                    if text_data[line[0]] == 'N/A':
                        text_data[line[0]] = line[5].replace('\n', '').replace('\r', '')
                    else:
                        text_data[line[0]] += ' >< ' + line[5].replace('\n', '').replace('\r', '')
    # before 1997
    if year == '1997':
        with open(text_root + '1996.txt', 'rb') as d:
            d.readline()
            while True:
                line = d.readline().decode('utf8', 'ignore')
                if not line:
                    break
                else:
                    line = line.split('|')
                    if line[0] in text_data.keys():
                        if text_data[line[0]] == 'N/A':
                            text_data[line[0]] = line[5].replace('\n', '').replace('\r', '')
                        else:
                            text_data[line[0]] += ' >< ' + line[5].replace('\n', '').replace('\r', '')
        with open(text_root + 'thru1995.txt', 'rb') as d:
            d.readline()
            while True:
                line = d.readline().decode('utf8', 'ignore')
                if not line:
                    break
                else:
                    line = line.split('|')
                    if line[0] in text_data.keys():
                        if text_data[line[0]] == 'N/A':
                            text_data[line[0]] = line[5].replace('\n', '').replace('\r', '')
                        else:
                            text_data[line[0]] += ' >< ' + line[5].replace('\n', '').replace('\r', '')

    # append text to filtered_data
    for entry in filtered_data:
        entry.append(text_data[entry[1]])

    return filtered_data


MDR_overall = set()

for i in range(1998, 2019):
    filtered_data, summary = interpret_dev(str(i))  # apply filter
    filtered_data = interpret_text(filtered_data, str(i))  # add text to filtered_data

    MDR = set()

    for entry in filtered_data:
        if 'DA VINCI' in entry[5] or 'DAVINCI' in entry[5]:
            MDR.add(entry[1])
            MDR_overall.add(entry(1))

    print(i, len(MDR))

print('Overall', len(MDR_overall))


