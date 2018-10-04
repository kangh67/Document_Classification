# Apply classfiers on MAUDE 2017
import re

# MAUDE 2017 original data
dev17 = 'data/foidev2017.txt'
text17 = 'data/foitext2017.txt'
mdr17 = 'data/mdrfoiThru2017.txt'


# Inclusion and exclusion keywords
in_gen = 'data/inclusion_generic.txt'
ex_gen = 'data/exclusion_generic.txt'
in_manu = 'data/inclusion_manu.txt'


# MAUDE 2017 filtered data
dev17_filtered = 'data/dev_2017_filtered.txt'
text17_filtered = 'data/text_2017_filtered.txt'


# MAUDE 2017 tsv file for classifier
maude_2017 = 'data/MAUDE_2017_noLabel.tsv'


# Initialization
dev = dict()
dev_count = 0
dev_brand = dict()
dev_generic = dict()
dev_manu = dict()


# Summary of MAUDE 2017
with open(dev17, 'rb') as d:
    line = d.readline()
    # print(line.decode('utf-8').split('|')[7])
    while True:
        line = d.readline()
        if not line:
            break
        else:
            l = line.decode('utf8', 'ignore').split('|')
            dev.setdefault(l[0], 0)
            dev_brand.setdefault(l[6], 0)   # brand name
            dev_generic.setdefault(l[7], 0)  # generic name
            dev_manu.setdefault(l[8], 0)    # manufacturer name
            dev[l[0]] += 1
            dev_brand[l[6]] += 1
            dev_generic[l[7]] += 1
            dev_manu[l[8]] += 1
            dev_count += 1

print('=== Summary of device data ===')
print('Unique MDR_REPORT_KEY: ', len(dev))
print('Unique BRAND_NAME:', len(dev_brand))
print('Unique GENERIC_NAME:', len(dev_generic))
print('Unique MANUFACTURER_D_NAME:', len(dev_manu))

dev_generic_sorted = sorted(dev_generic.items(), reverse=True, key=lambda kv: kv[1])
print('Ranking of generic names:', dev_generic_sorted)


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
dev_filtered_generic = dict()
dev_filtered_manu = dict()


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
            # Count generic name hit
            if generic in dev_filtered_generic.keys():
                dev_filtered_generic[generic] += 1
            else:
                dev_filtered_generic[generic] = 1
    for keyword in inclusion_manu.keys():
        if re.search(r'\b' + keyword + r'\b', manu):
            inclusion_manu[keyword] += 1
            flag_in_manu = True
            # Count manu name hit
            if manu in dev_filtered_manu.keys():
                dev_filtered_manu[manu] += 1
            else:
                dev_filtered_manu[manu] = 1

    return flag_in_ge or flag_in_manu


# Apply filter to dev 2017
print('=== Apply the filter on MAUDE 2017 data ===')
dev_filtered_count = 0
dev_filtered_mdr = set()
with open(dev17, 'rb') as d:
    line = d.readline()
    w = open(dev17_filtered, 'w')
    w.write(line.decode('utf8', 'ignore'))
    while True:
        line = d.readline()
        if not line:
            break
        else:
            line = line.decode('utf8', 'ignore')
            if apply_filter(line):
                dev_filtered_count += 1
                w.write(line)
                dev_filtered_mdr.add(line.split('|')[0])
    w.close()


# Count the activated keywords
in_gen_count = 0
for keyword in inclusion_generic.keys():
    if inclusion_generic[keyword] != 0:
        in_gen_count += 1
ex_gen_count = 0
for keyword in exclusion_generic.keys():
    if exclusion_generic[keyword] != 0:
        ex_gen_count += 1
in_manu_count = 0
for keyword in inclusion_manu.keys():
    if inclusion_manu[keyword] != 0:
        in_manu_count += 1


# Keyword hit summary after applying the filter
print('Dev record # after applying the filter:', dev_filtered_count, '(', len(dev_filtered_mdr), 'unique)')

inclusion_generic_sorted = sorted(inclusion_generic.items(), reverse=True, key=lambda kv: kv[1])
print('Hit # of generic inclusion keywords:', in_gen_count)
print('Hit ranking of generic inclusion keywords:', inclusion_generic_sorted)

exclusion_generic_sorted = sorted(exclusion_generic.items(), reverse=True, key=lambda kv: kv[1])
print('Hit # of generic exclusion keywords', ex_gen_count)
print('Hit ranking of generic exclusion keywords:', exclusion_generic_sorted)

inclusion_manu_sorted = sorted(inclusion_manu.items(), reverse=True, key=lambda kv: kv[1])
print('Hit # of manufacturer inclusion keywords', in_manu_count)
print('Hit ranking of manufacturer keywords:', inclusion_manu_sorted)

dev_filtered_generic_sorted = sorted(dev_filtered_generic.items(), reverse=True, key=lambda kv: kv[1])
print('Unique Generic name # of filtered data:', len(dev_filtered_generic))
print('Generic name ranking of filtered data:', dev_filtered_generic_sorted)

dev_filtered_manu_sorted = sorted(dev_filtered_manu.items(), reverse=True, key=lambda kv: kv[1])
print('Unique Manufacturer name # of filtered data:', len(dev_filtered_manu))
print('Manufacturer name ranking of filtered data:', dev_filtered_manu_sorted)


# Apply filtered MDR on text data
text_filtered_count = 0
text_filtered_mdr = set()
with open(text17, 'rb') as d:
    line = d.readline().decode('utf8', 'ignore')
    w = open(text17_filtered, 'w')
    w.write(line)
    while True:
        line = d.readline()
        if not line:
            break
        else:
            line = line.decode('utf8', 'ignore')
            if line.split('|')[0] in dev_filtered_mdr:
                text_filtered_count += 1
                text_filtered_mdr.add(line.split('|')[0])
                w.write(line)
    w.close()

print('Text record # after applying the filter:', text_filtered_count, '(', len(text_filtered_mdr), 'unique)')
print('Filtered dev file was saved at:', dev17_filtered)
print('Filtered text file was saved at:', text17_filtered)


# Create tsv file
text_2017 = dict()
with open(text17_filtered, 'r') as d:
    line = d.readline()
    while True:
        line = d.readline()
        if not line:
            break
        else:
            line = line.split('|')
            if line[0] in text_2017.keys():
                text_2017[line[0]] += ' >< ' + line[5].replace('\n', '')
            else:
                text_2017[line[0]] = line[5].replace('\n', '')

print('=== Create .tsv file ===')
with open(maude_2017, 'w') as w:
    w.write('MDR\tTEXT\n')
    for key in text_2017.keys():
        w.write(key + '\t' + text_2017[key] + '\n')
print('Saved at', maude_2017)
