import os
import sys
import copy

class ConvertAnnotation(object):
    def __init__(self, char_dir, origin_annotation_dir, split_rate=None):
        self.char_2_id = {}
        self.id_2_char = {}
        with open(char_dir, 'r') as fp:
            offset = 0
            for i, line in enumerate(fp):
                if line[0] in self.char_2_id.keys():
                    # print(line[0])
                    offset += 1
                else:
                    self.char_2_id[line[0]] = i - offset
                    self.id_2_char[i - offset] = line[0]
        num_class_id = len(self.char_2_id) - 1
        self.image_dir_list = []
        self.image_annotation = []
        with open(origin_annotation_dir, 'r') as fp:
            for i, line in enumerate(fp):
                line = line.strip().split(' ')
                assert len(line) == 2
                self.image_dir_list.append(line[0])
                self.image_annotation.append(line[1])
        self.image_id_annotation = []
        for i, annotation in enumerate(self.image_annotation):
            id_tmp = []
            for _, wd in enumerate(annotation):
                if wd in self.char_2_id.keys():
                    id_tmp.append(self.char_2_id[wd])
                else:
                    num_class_id += 1
                    self.char_2_id[wd] = num_class_id
                    self.id_2_char[num_class_id] = wd
                    id_tmp.append(num_class_id)
            self.image_id_annotation.append(id_tmp)
        if split_rate:
            split_point = int(len(self.image_id_annotation) * (1-split_rate))
            self.image_id_annotation_train = self.image_id_annotation[:split_point]
            self.image_id_annotation_test = self.image_id_annotation[split_point:]
            self.image_dir_list_train = self.image_dir_list[:split_point]
            self.image_dir_list_test = self.image_dir_list[split_point:]
            del self.image_id_annotation
            del self.image_dir_list

    def generate_new_chart(self, save_dir):
        with open(save_dir, 'w') as fp:
            for i, wd in enumerate(self.char_2_id.keys()):
                fp.write(wd + '\n')

class MergeDict(object):
    def __init__(self, first_dict, second_dict):
        self.first_char_2_id = {}
        self.second_char_2_id = {}
        with open(first_dict, 'r') as fp:
            first_offset = 0
            for i, line in enumerate(fp.readlines()):
                if line in self.first_char_2_id.keys():
                    first_offset += 1
                else:
                    self.first_char_2_id[line[0]] = i - first_offset
        # print(len(self.first_char_2_id))
        with open(second_dict, 'r') as fp:
            second_offset = 0
            for i, line in enumerate(fp.readlines()):
                if line in self.second_char_2_id.keys():
                    second_offset += 1
                else:
                    self.second_char_2_id[line[0]] = i - second_offset
        # print(len(self.second_char_2_id))
        first_num_class_id = len(self.second_char_2_id) - 1
        # num_unkown = 0
        for i, wd in enumerate(self.second_char_2_id):
            if wd in self.first_char_2_id.keys():
                pass
            else:
                # num_unkown += 1
                first_num_class_id += 1
                self.first_char_2_id[wd] = first_num_class_id
        print(len(self.first_char_2_id))


    def generate_new_dict(self, save_dir):
        with open(save_dir, 'w') as fp:
            for i, wd in enumerate(self.first_char_2_id.keys()):
                if i == len(self.first_char_2_id):
                    fp.write(wd)
                else:
                    fp.write(wd + '\n')



if __name__ == '__main__':
    chart_dir = '/home/oushu/tongli/ocr/ocr/TextGen/TextRecognitionDataGenerator/newcorpus1/Train/c1train_labels.txt'
    # annotation_dir = '/home/oushu/lixingyu/DataSet/image_chinese_dataset/labels.txt'
    # CA = ConvertAnnotation(chart_dir, annotation_dir, 0.05)
    # CA.generate_new_chart('/home/oushu/lixingyu/DataSet/new_char.txt')
    second_chart_dir = '/home/oushu/lixingyu/DataSet/char_6112.txt'
    MD = MergeDict(chart_dir, second_chart_dir)
    MD.generate_new_dict('/home/oushu/lixingyu/DataSet/new_char.txt')