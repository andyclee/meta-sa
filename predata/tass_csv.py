import os
import csv
import xml.etree.ElementTree as ET

data_dir = 'TASSData'

for fn in os.listdir(data_dir):
    print('processing', fn)
    
    csv_lines = []

    fname, ext = fn.split('.')
    xml_tree = ET.parse(os.path.join(data_dir, fn))
    for twt in xml_tree.getroot():
        twt_content = twt.find('content').text
        twt_sent = twt.find('sentiment')[0][0].text
        csv_lines.append( [twt_content, twt_sent] )

    _, _, country, setname = fname.split('_')
    with open('es-' + country + '_' + setname + '.csv', 'w+') as cf:
        csv_writer = csv.writer(cf, delimiter=',')
        for cline in csv_lines:
            csv_writer.writerow(cline)

