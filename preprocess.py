import os
import csv
import re
import string

data_dir = 'test_data'

url_regex = re.compile(
    r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:'".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
)

email_regex = re.compile(
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
)

phone_regex = re.compile(
    r'\b(?:\+\d{1,2}\s?)?(?:\(\d{1,4}\)|\d{1,4})[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b'
)

def clean_content(txt):

    # Clean the text of a content
    
    # Remove URLs
    #no_http = http_regex.sub('', txt)
    #no_www = www_regex.sub('', no_http)
    #no_url = com_regex.sub('', no_www)
    no_url = url_regex.sub('', txt)
    no_email = email_regex.sub('', no_url)
    no_phone = phone_regex.sub('', no_email)

    # Strip punctuation
    no_punc = no_phone.translate(str.maketrans('', '', string.punctuation))
    no_extra_space = re.sub(' +', ' ', no_punc)

    final = no_extra_space.strip().lower()

    return final

def map_sentiment(sent, sentmap):

    # -1 for negative, 0 for neutral, 1 for positive
    if sent in sentmap:
        return sentmap[sent]
    return None

def process(fname, lang, setname, sentmap):

    # Process the data at fname

    out_fn = 'clean_{l}_{s}.csv'.format(l=lang, s=setname)
    out_fo = open(os.path.join(data_dir, out_fn), 'w+') 
    out_writer = csv.writer(out_fo, delimiter=',')

    with open(fname, 'r') as fcsv:
        csvreader = csv.reader(fcsv)
        for row in csvreader:
            clean_text = clean_content(row[0])
            sent_lbl = map_sentiment(row[1], sentmap)

            # Skip labels like NONE
            if sent_lbl is None:
                continue
            out_writer.writerow( [ clean_text, sent_lbl ] )

    out_fo.close()

def proc_dir(dirname, sentmap):
    for ef in os.listdir(dirname):
        if ef == '.placeholder':
            continue
        print('cleaning', ef)
        lang_set, ext = ef.split('.')
        lang, dset = lang_set.split('_')
        process(os.path.join(dirname, ef), lang, dset, sentmap)

if __name__ == '__main__':

    # Start with English data
    eng_dir = os.path.join(data_dir, 'en')
    eng_sentmap = { '0' : -1, '2' : 0, '4' : 1 }
    proc_dir(eng_dir, eng_sentmap)

    # Now the Spanish data
    es_dir = os.path.join(data_dir, 'es')
    es_sentmap = { 'N' : -1, 'NEU' : 0, 'P' : 1 }
    #proc_dir(es_dir, es_sentmap)

    # Now the African data
    afr_dir = os.path.join(data_dir, 'afr')
    afr_sentmap = { 'negative' : -1, 'neutral' : 0, 'positive' : 1 }
    #proc_dir(afr_dir, afr_sentmap)

