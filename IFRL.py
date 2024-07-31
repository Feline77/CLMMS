from tqdm import tqdm
import random

def RunLengthEncoding(message):
    compressed_message = []
    count = 1
    for i in tqdm(range(0, len(message))):
        if i + 1 < len(message) and message[i] == message[i + 1]:
            count += 1
        else:
            compressed_message.append(message[i])
            compressed_message.append(count)
            count = 1
    return compressed_message

def RunLengthDecoding(compressed_message):
    decompressed_message = []
    for i in tqdm(range(0, len(compressed_message), 2)):
        char = compressed_message[i]
        count = compressed_message[i + 1]
        decompressed_message += [char] * count
    return decompressed_message

def SecondEncoding(init_compress_data):
    total_compress_data = []
    for i in tqdm(range(0, len(init_compress_data), 2)):

        # 转化value
        value = init_compress_data[i]

        if value == 1:
            value = random.randint(50, 63) * 2 + 1
        elif value == 0:
            value = random.randint(50, 63) * 2

        # 转化count，以2字符为断点分割
        count = str(init_compress_data[i + 1])  # "10096","1782","100023"
        count_length = len(count)

        # 次数大于99次，即当字符串长度大于2的时候
        if count_length > 2:
            slices = count_length // 2
            trans_count = [count[i * 2:(i + 1) * 2] for i in range(slices)]
            if slices * 2 < count_length:
                trans_count.append(count[slices * 2:count_length])

            # 检查trans_count中是否有以0开头的字符：
            for flag, item in enumerate(trans_count):
                if len(item) > 1:
                    if item[0] == "0":
                        check_trans_count = trans_count[:flag] + [item[0], item[1]] + trans_count[flag + 1:]
                        trans_count = check_trans_count


        elif count_length <= 2:
            trans_count = [count]

        # 将trans_count转换成int
        trans_count_ = [int(i) for i in trans_count]

        total_compress_data.append([value] + trans_count_)

        # 铺平compress_data
        compress_data = []
        for valuecount in total_compress_data:
            compress_data += [i for i in valuecount]

    return compress_data

def SecondDecoding(compress_data):
    # 获取每段value&count的索引
    value_index = []
    for index, cand in enumerate(compress_data):
        if cand > 99:
            value_index.append(index)
    value_index.append(len(compress_data))

    decompress_data = []

    for flag in tqdm(range(len(value_index[:-1]))):
        index = value_index[flag]
        start = value_index[flag] + 1
        end = value_index[flag + 1]

        value = compress_data[index]
        if value % 2 == 0:
            value = 0
        elif value % 2 == 1:
            value = 1

        trans_count = compress_data[start:end]
        trans_count_ = [str(i) for i in trans_count]
        trans_count = ""
        for i in trans_count_:
            trans_count += i
        trans_count = int(trans_count)

        decompress_data += [value, trans_count]

    return decompress_data

def ImprovedRunLengthEncoding(bit_streams):
    init_compress = RunLengthEncoding(bit_streams)
    fina_compress = SecondEncoding(init_compress)
    return fina_compress

def ImprovedRunLengthDecoding(fina_compress):
    init_comresss = SecondDecoding(fina_compress)
    decompress_data = RunLengthDecoding(init_comresss)
    return decompress_data