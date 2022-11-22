# -*- coding: utf-8 -*-
# @Author: Jie Zhou
# @Time: 2018/12/2 下午1:39
# @Project: unified_chinese_multi_tasking_framework
# @File: ctb_preprocess.py
# @Software: PyCharm

import os
import errno
import nltk
import sys
from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader
import argparse

penn2malt_dir = '~/Downloads/Penn2Malt.jar'


def con2dep(con_path):
    '''java -jar Penn2Malt.jar chtbtest.txt chn_headrules.txt 3 2 chtb'''
    cmd_str = 'java -jar ' + penn2malt_dir + ' ' + con_path +' ' + '~/Downloads/chn_headrules.txt 3 2 chtb'
    os.system(cmd_str)


def remove_xml_for_ctb5(ctb_dir, out_dir):
    check_path(out_dir)
    ctb_dir = os.path.join(ctb_dir, 'bracketed')
    f_list = [f for f in os.listdir(ctb_dir) if os.path.isfile(os.path.join(ctb_dir, f)) and f.startswith('chtb')]
    for f in f_list:
        with open(os.path.join(ctb_dir, f), encoding='utf-8') as origin, open(os.path.join(out_dir, f), 'w') as clean_data:
            sentence_flag = False
            try:
                for l in origin:
                    if l.startswith('<S ID='):
                        sentence_flag = True
                    elif l.startswith('</S>'):
                        sentence_flag = False
                    elif sentence_flag:
                        clean_data.write(l)
            except:
                pass
    return


def remove_xml_for_ctb8(ctb_dir, out_dir):
    check_path(out_dir)
    ctb_dir = os.path.join(ctb_dir, 'bracketed')
    f_list = [f for f in os.listdir(ctb_dir) if os.path.isfile(os.path.join(ctb_dir, f)) and f.startswith('chtb')]
    for f in f_list:
        with open(os.path.join(ctb_dir, f), encoding='utf-8') as origin, open(os.path.join(out_dir, f), 'w') as clean_data:
            for l in origin:
                if not l.startswith('<') and l is not '\n':
                    clean_data.write(l)
    return


def combine_id_list(f_id_list, out_dir, task, ctb_nltk_dir):
    print('Combining ' + out_dir)
    file_names = []
    for f_id in f_id_list:
        f = ''
        if f_id <= 931 or (f_id >= 4000 and f_id <= 4050):
            f = 'chtb_%04d.nw' % f_id
        elif f_id >= 1001 and f_id <= 1151:
            f = 'chtb_%04d.mz' % f_id
        elif (f_id >= 2000 and f_id <= 3145) or (f_id >= 4051 and f_id <= 4111):
            f = 'chtb_%04d.bn' % f_id
        elif f_id >= 4112 and f_id <= 4197:
            f = 'chtb_%04d.bc' % f_id
        elif f_id >= 4198 and f_id <= 4411:
            f = 'chtb_%04d.wb' % f_id
        elif f_id >= 5000 and f_id <= 5558:
            f = 'chtb_%04d.df' % f_id
        if os.path.isfile(os.path.join(ctb_nltk_dir, f)):
            file_names.append(f)
    with open(out_dir, 'w') as out:
        combine_file(file_names, out, ctb, task, add_s=True)


def combine_file(f_id_list, out_dir, treebank, task, add_s=False):
    print('%d files...' % len(f_id_list))
    total_s = 0
    if task == 'parse_for_dep':
        for n, f_id in enumerate(f_id_list):
            if n % 10 == 0 or n == len(f_id_list) - 1:
                print("%c%.2f%%" % (13, (n + 1) / float(len(f_id_list)) * 100), end='')
            with open(os.path.join(ctb_nltk_dir, f_id), encoding='utf-8') as origin:
                out_dir.writelines(origin.readlines())
        return
    for n, f_id in enumerate(f_id_list):
        if n % 10 == 0 or n == len(f_id_list) - 1:
            print("%c%.2f%%" % (13, (n + 1) / float(len(f_id_list)) * 100), end='')
        sen_list = treebank.parsed_sents(f_id)
        for s in sen_list:
            if task == 'bracketed':
                if add_s:
                    out_dir.write('(S {})'.format(s.pformat(margin=sys.maxsize)))
                else:
                    out_dir.write(s.pformat(margin=sys.maxsize))
            elif task == 'pos':
                for word, tag in s.pos():
                    if tag == '-NONE-':
                        continue
                    out_dir.write('{}\t{}\n'.format(word, tag))
            elif task == 'pos-pku':
                for word, tag in s.pos():
                    if tag == '-NONE_':
                        continue
                    out_dir.write('{}\{} '.format(word, tag))
            elif task == 'seg':
                for word, tag in s.pos():
                    if tag == '-NONE-':
                        continue
                    if len(word) == 1:
                        out_dir.write(word + '\tS\n')
                    else:
                        out_dir.write(word[0] + '\tB\n')
                        for w in word[1:len(word) - 1]:
                            out_dir.write(w + '\tM\n')
                        out_dir.write(word[-1] + '\tE\n')
            elif task == 'st':
                for word, tag in s.pos():
                    if tag == '-NONE-':
                        continue
                    if len(word) == 1:
                        out_dir.write('{}\tS-{}\n'.format(word, tag))
                    else:
                        out_dir.write('{}\tB-{}\n'.format(word[0], tag))
                        for w in word[1:len(word) - 1]:
                            out_dir.write('{}\tM-{}\n'.format(w, tag))
                        out_dir.write('{}\tE-{}\n'.format(word[-1], tag))
            else:
                raise RuntimeError('Invalid task: {}'.format(task))
            out_dir.write('\n')
            total_s += 1
    print('\n%d sentences.\n' % total_s)


def check_path(path):
    try:
        os.makedirs(path)
    except OSError as expection:
        if expection.errno is not errno.EEXIST:
            raise
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocess Chinese Treebank into train/dev/test sets')
    parser.add_argument('--ctb-v', required=True, default=5, help='The version of Chinese Treebank.')
    parser.add_argument('--origin-ctb', required=True, help='The dir of origin corpus')
    parser.add_argument('--output-dir', required=True,
                        default='~/Desktop/ctb_corpus/', help='The dir of train/dev/test sets.')
    parser.add_argument('--task', default='bracketed',
                        help='The part of chinese treebank to generate the train/dev/test sets.')
    options = parser.parse_args()
    task = options.task
    out_d = options.output_dir
    ctb_d = options.origin_ctb
    ctb_v = int(options.ctb_v)
    out_d = out_d + 'ctb' + str(ctb_v) + '/'
    check_path(out_d)

    ctb_nltk_dir = None

    for tmp_dir in nltk.data.path:
        if os.path.isdir(tmp_dir):
            ctb_nltk_dir = tmp_dir

    if ctb_nltk_dir is None:
        nltk.download('ptb')
        for tmp_dir in nltk.data.path:
            if os.path.isdir(tmp_dir):
                ctb_nltk_dir = tmp_dir

    ctb_nltk_dir = os.path.join(ctb_nltk_dir, 'corpora')
    ctb_nltk_dir = os.path.join(ctb_nltk_dir, 'ctb')

    print('Removing xml tags...')
    if ctb_v == 5:
        remove_xml_for_ctb5(ctb_d, ctb_nltk_dir)
    else:
        remove_xml_for_ctb8(ctb_d, ctb_nltk_dir)

    print('Importing into nltk...')
    ctb = LazyCorpusLoader('ctb', BracketParseCorpusReader, r'chtb_.*\.*', tagset='unknown')
    if ctb_v == 5:
        '''
            ctb 5.0 cws&pos
            Sections
            Sentences
            1–270
            400–931
            1001–1151
            18,085 493,892
            dev 301–325 350 6821
    
            test 271–300 348 8008
            
            ctb 5.0 dependency
            train 1-815 1001-1136
            dev 886-931 1148-1151
            test 816-885 1137-1147
        '''
        if task == 'bracketed' or task == 'parse_for_dep':
            train_data_id = list(range(1, 815 + 1)) + list(range(1001, 1136 + 1))
            dev_data_id = list(range(886, 931 + 1)) + list(range(1148, 1151 + 1))
            test_data_id = list(range(816, 885 + 1)) + list(range(1137, 1147 + 1))
            file_type = 'txt'
        elif task == 'seg' or task == 'pos' or task == 'st':
            train_data_id = list(range(1, 270 + 1)) + list(range(400, 1151 + 1))
            dev_data_id = list(range(301, 325 + 1))
            test_data_id = list(range(271, 300 + 1))
            file_type = 'tsv'
        else:
            raise RuntimeError('Invalid task: {}'.format(task))
    elif ctb_v == 6:
        '''
            ctb6.0
            Data set
            CTB chapter IDs
            train
            81-325, 400-454, 600-885, 900, 500-554, 590-596, 1001-1017, 1019, 1021-1035, 1037-1043,
             1045-1059, 1062-1071, 1073-1078, 1100-1117, 1130-1131, 1133-1140, 1143-1147, 1149-1151, 
             2000-2139, 2160-2164, 2181-2279, 2311-2549, 2603-2774, 2820-3079, 
            Dev
            41-80, 1120-1129, 2140-2159, 2280-2294, 2550-2569, 2775-2799, 3080-3109, 
            Test
            1-40,901-931, 1018, 1020, 1036, 1044, 1060-1061, 1072, 1118-1119, 1132, 1141-1142, 1148, 
            2165-2180, 2295-2310, 2570-2602, 2800-2819, 3110-3145
            
        '''
        train_data_id = list(range(81, 325 + 1)) + list(range(400, 454 + 1)) + list(range(600, 885 + 1)) + [900]\
                        + list(range(500, 554)) + list(range(590, 596 + 1)) \
                        + list(range(1001, 1017 + 1)) + list(range(1019, 1019 + 1)) + list(range(1021, 1035 + 1)) \
                        + list(range(1037, 1043 + 1)) + list(range(1045, 1059 + 1)) + list(range(1062, 1071 + 1)) \
                        + list(range(1073, 1078 + 1)) + list(range(1100, 1117 + 1)) + list(range(1130, 1131 + 1))\
                        + list(range(1133, 1140 + 1)) + list(range(1143, 1147 + 1)) + list(range(1149, 1151 + 1))\
                        + list(range(2000, 2139 + 1)) + list(range(2160, 2164 + 1)) + list(range(2181, 2279 + 1))\
                        + list(range(2311, 2549 + 1)) + list(range(2603, 2774 + 1)) + list(range(2820, 3079 + 1))
        dev_data_id = list(range(41, 80 + 1)) + list(range(1120, 1129 + 1)) + list(range(2140, 2159 + 1)) \
                      + list(range(2280, 2294 + 1)) + list(range(2550, 2569 + 1)) + list(range(2775, 2799 + 1)) \
                      + list(range(3080, 3109 + 1))
        test_data_id = list(range(1, 40 + 1)) + list(range(901, 931 + 1)) + list(range(1018, 1018 + 1))\
                       + list(range(1020, 1020 + 1)) + list(range(1036, 1036 + 1)) + list(range(1044, 1044 + 1))\
                       + list(range(1060, 1061 + 1)) \
                       + list(range(1072, 1072 + 1)) + list(range(1118, 1119 + 1)) + list(range(1132, 1132 + 1)) \
                       + list(range(2165, 2180 + 1)) + list(range(2295, 2310 + 1)) + list(range(2570, 2602 + 1)) \
                       + list(range(2800, 2819 + 1)) + list(range(1141, 1142 + 1)) + list(range(1148, 1148 + 1)) \
                       + list(range(3110, 3145 + 1))
        file_type = 'txt'
    elif ctb_v == 7:
        '''
            ctb7.0
            Data set
            CTB chapter IDs
            train
            81-143, 175-202, 234-270, 429-454, 500-554, 590, 593-596, 600-612, 618-642, 674-885,
             1001-1008, 1021, 1037-1043, 1045-1059, 1062-1071, 1073-1117, 1130-1131, 1133-1140,
             1143-1147, 1149-1151, 2011-2109, 2221-2269, 2331-2509, 2641-2759, 2846-3039, 4000-4029,
             4071-4083, 4088-4089, 4092-4095, 4098-4105, 4116-4117, 4122-4126, 4129-4131, 4140-4157,
             4170-4187, 4190-4195, 4262-4320, 4323-4333, 4338-4399
            Dev
            41-80,203-233,301-325,400-409,591,613-617,643-673,
            1022-1035,1120-1129,2110-2159,2270-2294,2510-2569,
            2760-2799,3040-3109,4040-4059,4084-4085,4090,4096,
            4106-4108,4113- 4115,4121,4128,4132,4135,4158-4162,4169,4189,4196,
            4236-4261,4322,4335-4336,4407-4411
            Test
            1-40,144-174,271-300,410-428,592,900-931,1009-1020,1036,1044,
            1060-1061,1072,1118-1119,1132,1141- 1142,1148,2000-2010,
            2160-2220,2295-2330,2570-2640,2800-2845,3110-3145,4030-4039,
            4060- 4070,4086-4087,4091,4097,4109-4112,4118-4120,4127,4133-4134,
            4136-4139,4163-4168,4188,4197- 4235,4321,4334,4337,4400-4406
        '''
        train_data_id = list(range(81, 143 + 1)) + list(range(175, 202 + 1)) + list(range( 234, 270 + 1)) \
                        + list(range(429, 454 + 1)) + list(range(500, 554 + 1)) + list(range(590, 590 + 1)) \
                        + list(range(593, 596 + 1)) + list(range(600, 612 + 1)) + list(range(674, 885 + 1)) \
                        + list(range( 1001, 1008 + 1)) + list(range( 1021, 1021 + 1)) + list(range( 1037, 1043 + 1))\
                        + list(range( 1045, 1059 + 1)) + list(range( 1062, 1071 + 1)) + list(range( 1073, 1117 + 1))\
                        + list(range( 1130, 1131 + 1)) + list(range( 1133, 1140 + 1)) + list(range( 1143, 1147 + 1))\
                        + list(range( 1149, 1151 + 1)) + list(range( 2011, 2109 + 1)) + list(range( 2221, 2269 + 1))\
                        + list(range( 2311, 2509 + 1)) + list(range( 2641, 2759 + 1)) + list(range( 2846, 3039 + 1))\
                        + list(range( 4000, 4029 + 1)) + list(range( 4071, 4083 + 1)) + list(range( 4088, 4089 + 1))\
                        + list(range( 4092, 4095 + 1)) + list(range(4098, 4105 + 1)) + list(range( 4116, 4117 + 1))\
                        + list(range( 4122, 4126 + 1)) + list(range( 4129, 4131 + 1)) + list(range( 4140, 4157 + 1)) \
                        + list(range( 4170, 4187+ 1)) + list(range( 4190, 4195 + 1)) + list(range( 41262, 4320 + 1)) \
                        + list(range(4323, 4333 + 1)) + list(range( 4338, 4399 + 1))
        dev_data_id = list(range(41, 80 + 1)) + list(range( 203, 233 + 1))  + list(range(301, 325 + 1))\
                      + list(range( 2110, 2159 + 1)) + list(range( 400, 409 + 1)) + list(range(591, 591 + 1))\
                      + list(range( 613, 617 + 1)) + list(range( 643, 673 + 1)) + list(range(1022, 1035 + 1))\
                      + list(range( 1120, 1129 + 1)) + list(range( 2270, 2294 + 1)) + list(range(2510, 2569 + 1))\
                      + list(range( 2760, 2799 + 1)) + list(range( 3040, 3109 + 1)) + list(range( 4040, 4059 + 1))\
                      + list(range( 4084, 4085 + 1)) + list(range(4090, 4090 + 1)) + list(range(4096, 4096 + 1)) \
                      + list(range(4106, 4108 + 1)) + list(range(4113, 4115 + 1)) + [4121,4128,4132,4135]\
                      + list(range(4158, 4162 + 1)) + [4169,4189,4196] + list(range(4236, 4261 + 1)) + [4322]\
                      + list(range(4335, 4336 + 1)) + list(range( 4407, 4411 + 1))
        test_data_id = list(range(1, 40 + 1)) + list(range( 144, 174 + 1)) + list(range( 271, 300 + 1))\
                       +list(range(401, 428 + 1)) + [592] + list(range( 900, 931 + 1)) + list(range(1009, 1020 + 1))\
                       + list(range( 1036, 1036 + 1)) + list(range( 1044, 1044 + 1)) + list(range( 1060, 1061 + 1))\
                       + list(range( 1072, 1072 + 1)) + list(range( 1118, 1119 + 1)) + list(range( 1132, 1132 + 1))\
                       + list(range(2000, 2010 + 1)) + list(range( 2160, 2220 + 1)) + list(range( 2295, 2330 + 1))\
                       + list(range( 2570, 2640 + 1)) + list(range( 2800, 2845 + 1)) + list(range( 1141, 1142 + 1))\
                       + list(range( 1148, 1148 + 1)) + list(range( 3110, 3145 + 1)) + list(range(4030, 4039 + 1))\
                       + list(range(4060, 4070 + 1)) + list(range(4086, 4087)) + list(range(4109, 4112 + 1))\
                       + list(range(4136, 4139 + 1)) + list(range(4118, 4120 + 1)) + list(range(4133, 4134 + 1))\
                       + list(range(4163, 4168 + 1)) + list(range(4197, 4235 + 1)) + [4091, 4097, 4127, 4188, 4321, 4334, 4337]\
                       + list(range(4400, 4406 + 1))
        file_type = 'txt'
    elif ctb_v == 8:
        '''
        Dataset CTB8.0 chapter IDs
         Train
            81-0143, 0170-0270, 0400-0899, 1001-1017, 1019, 1021-1035,
            1037- 1043, 1045-1059, 1062-1071, 1073- 1117, 1120-1131,
            1133-1140, 1143- 1147, 1149-1151, 2000-2139, 2160-2164, 2181-2279, 2311-2549, 2603-2774, 2820-2915,
            4000- 4099, 4112-4180, 4198-4368, 5000- 5446

        Dev 
            44-80, 0301-0326, 2140-2159, 2280-2294, 2550-2569, 2775-2799
             2916-3030, 4100-4106, 4181-4189, 4369-4390, 5447-5492

        Test
            0001-0043, 0144-0169, 0271-0301, 0900-0931,
            1018, 1020, 1036, 1044, 1060, 1061, 1072, 1118, 1119, 1132,
            1141, 1142, 1148, 2165-2180, 2295-2310, 2570-2602, 2800-2819, 
            3031-3145, 4107- 4111, 4190-4197, 4391-4411, 5493- 5558
        '''
        train_data_id = list(range(81, 143 + 1)) + list(range(170, 270 + 1)) + list(range( 400, 899 + 1))\
                        + list(range( 1001, 1017 + 1)) + list(range( 1019, 1019 + 1)) + list(range( 1021, 1035 + 1))\
                        + list(range( 1037, 1043 + 1)) + list(range( 1045, 1059 + 1)) + list(range( 1062, 1071 + 1))\
                        + list(range( 1073, 1117 + 1)) + list(range( 1120, 1131 + 1)) + list(range( 1133, 1140 + 1))\
                        + list(range( 1143, 1147 + 1)) + list(range( 1149, 1151 + 1)) + list(range( 2000, 2139 + 1))\
                        + list(range( 2160, 2164 + 1)) + list(range( 2181, 2279 + 1)) + list(range( 2311, 2549 + 1))\
                        + list(range( 2603, 2774 + 1)) + list(range( 2820, 2915 + 1)) + list(range( 4000, 4099 + 1))\
                        + list(range( 4112, 4180 + 1)) + list(range( 4198, 4368 + 1)) + list(range( 5000, 5446 + 1))
        dev_data_id = list(range(44, 80 + 1)) + list(range( 301, 326 + 1)) + list(range( 2140, 2159 + 1)) \
                      + list(range( 2280, 2294 + 1)) + list(range( 2550, 2569 + 1)) + list(range( 2775, 2799 + 1))\
                      + list(range( 2916, 3030 + 1)) + list(range( 4100, 4106 + 1)) + list(range( 4181, 4189 + 1))\
                      + list(range( 4369, 3290 + 1)) + list(range( 5447, 5492 + 1))
        test_data_id = list(range(1, 43 + 1)) + list(range( 144, 169 + 1)) + list(range( 271, 301 + 1))\
                       + list(range( 900, 931 + 1)) + list(range( 1018, 1018 + 1)) + list(range( 1020, 1020 + 1))\
                       + list(range( 1036, 1036 + 1)) + list(range( 1044, 1044 + 1)) + list(range( 1060, 1061 + 1))\
                       + list(range( 1072, 1072 + 1)) + list(range( 1118, 1119 + 1)) + list(range( 1132, 1132 + 1))\
                       + list(range( 2165, 2180 + 1)) + list(range( 2295, 2310 + 1)) + list(range( 2570, 2602 + 1))\
                       + list(range( 2800, 2819 + 1)) + list(range( 1141, 1142 + 1)) + list(range( 1148, 1148 + 1))\
                       + list(range( 3031, 3145 + 1)) + list(range( 4107, 4111 + 1)) + list(range( 4190, 4197 + 1))\
                       + list(range( 4391, 4411 + 1)) + list(range( 5493, 5558 + 1))
        file_type = 'txt'
    combine_id_list(train_data_id, os.path.join(out_d, 'train.{}'.format(file_type)), task, ctb_nltk_dir)
    combine_id_list(dev_data_id, os.path.join(out_d, 'dev.{}'.format(file_type)), task, ctb_nltk_dir)
    combine_id_list(test_data_id, os.path.join(out_d, 'test.{}'.format(file_type)), task, ctb_nltk_dir)
    print('\nDone!\n')






