def get_primary_gb():
    start = 0xB0A1
    end = 0xD7FA
    gb_list = []
    gb_id = 0
    for i in range(start, end):
        if i & 0xF0 < 0xA0 or i & 0xFF == 0xA0 or i & 0xFF == 0xFF:
            continue
        character = str(i.to_bytes(length=2, byteorder='big'), 'gb2312')
        gb_list.append((character, gb_id))
        gb_id += 1
    return gb_list


def get_char_map():
    gb_list = get_primary_gb()
    label_id = {}
    id_label = {}
    for char, gb_id in gb_list:
        label_id[char] = gb_id
        id_label[gb_id] = char
    return label_id, id_label
