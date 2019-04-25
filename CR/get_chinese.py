def get_primary_gb():
    char_list = get_chinese()
    char_list.extend(get_english())
    char_list.extend(get_number())
    char_list.extend(get_punctuation())
    gb_list = [(ch, ch_id) for ch_id, ch in enumerate(char_list)]
    return gb_list


def get_punctuation():
    # gb_list = []
    # for i in range(0xA1A2, 0xA1FF):
    #     character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
    #     gb_list.append(character)
    #
    # for i in range(0xA3A1, 0xA3B0):
    #     character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
    #     gb_list.append(character)
    # for i in range(0xA3BA, 0xA3C1):
    #     character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
    #     gb_list.append(character)
    # for i in range(0xA3DB, 0xA3E1):
    #     character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
    #     gb_list.append(character)
    # for i in range(0xA3FB, 0xA3FF):
    #     character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
    #     gb_list.append(character)
    punctuation_list = '!@#$%^&*()_+~{}|:"<>?-=`[]\\;\',./' + '，。；‘’【】、！￥'
    return [char for char in punctuation_list]


def get_number():
    # gb_list = []
    # for i in range(0xA3B0, 0xA3BA):
    #     character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
    #     gb_list.append(character)
    # return gb_list
    return [str(num) for num in range(10)]


def get_english():
    # gb_list = []
    # for i in range(0xA3C1, 0xA3DB):
    #     character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
    #     gb_list.append(character)
    # for i in range(0xA3E1, 0xA3FB):
    #     character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
    #     gb_list.append(character)
    # return gb_list
    return [alpha for alpha in 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz']


def get_chinese():
    start = 0xB0A1
    end = 0xD7FA
    gb_list = []
    for i in range(start, end):
        if i & 0xF0 < 0xA0 or i & 0xFF == 0xA0 or i & 0xFF == 0xFF:
            continue
        character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
        gb_list.append(character)
    return gb_list


def get_char_map():
    gb_list = get_primary_gb()
    label_id = {}
    id_label = {}
    for char, gb_id in gb_list:
        label_id[char] = gb_id
        id_label[gb_id] = char
    return label_id, id_label


if __name__ == '__main__':
    # gb_list = get_primary_gb()
    # char_list = [char for char, _ in gb_list]
    # print(label_id)
    pass
