import numpy as np


# 获取用于im2col的索引
def get_im2col_indices(X_shape, kernel_H, kernel_W, stride, pad):
    # 获取输入和输出的形状
    _, channels, in_H, in_W = X_shape
    out_H = (in_H + 2 * pad - kernel_H) // stride + 1
    out_W = (in_W + 2 * pad - kernel_W) // stride + 1
  
    level1 = np.tile(np.repeat(np.arange(kernel_H), kernel_W), channels)
    everyLevels = stride * np.repeat(np.arange(out_H), out_W)
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    slide1 = np.tile(np.tile(np.arange(kernel_W), kernel_H), channels)
    everySlides = stride * np.tile(np.arange(out_W), out_H)
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    k = np.repeat(np.arange(channels), kernel_H * kernel_W).reshape(-1, 1)

    return i, j, k


# 用于将NCHW格式的批次图像转换为列并拼接为二维矩阵，可将卷积运算转换为矩阵乘法
def im2col(X, kernel_H, kernel_W, stride, pad):
    # 将输入图像X用0填充
    X_padded = np.pad(X, ((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant')
    # 获取用于im2col的索引
    i, j, k = get_im2col_indices(X.shape, kernel_H, kernel_W, stride, pad)
    # 将填充后的图像X按k,i,j拼接为列
    cols = X_padded[:, k, i, j]
    # 将拼接后的列按最后一维拼接
    cols = np.concatenate(cols, axis=-1)
    return cols


# 用于将列拼接的二维矩阵转换回NCHW格式的原矩阵
def col2im(input_grad_col, X_shape, kernel_H, kernrl_W, stride, pad):
    # 获取输入和输出的形状
    batch_size, channels, H, W = X_shape
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((batch_size, channels, H_padded, W_padded))

    # 获取用于im2col的索引
    i, j, k = get_im2col_indices(X_shape, kernel_H, kernrl_W, stride, pad)
    # 将输入梯度input_grad_col按batch_size,k,i,j拼接为列
    dX_col_reshaped = np.array(np.hsplit(input_grad_col, batch_size))
    # 将拼接后的列按k,i,j添加到填充后的图像X_padded中
    np.add.at(X_padded, (slice(None), k, i, j), dX_col_reshaped)

    return X_padded[:, :, pad:-pad, pad:-pad]


# 获取用于pool2row的索引
def get_pool2row_indices(X_shape, kernel_H, kernel_W, stride):
    _, channels, H, W = X_shape
    out_H = (H - kernel_H) // stride + 1
    out_W = (W - kernel_W) // stride + 1

    level1 = np.tile(stride * np.repeat(np.arange(out_H), out_W), channels)
    everyLevels = np.repeat(np.arange(kernel_H), kernel_W)

    slide1 = np.tile(np.arange(kernel_W), kernel_H)
    everySlides = stride * np.tile(np.arange(out_W), out_H * channels)

    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)
    j = slide1.reshape(1, -1) + everySlides.reshape(-1, 1)

    k = np.repeat(np.arange(channels), out_H * out_W).reshape(-1, 1)

    return i, j, k


# 将输入图像X按k,i,j拼接为行
def pool2row_indices(X, kernel_H, kernel_W, stride):
    # 获取用于pool2row的索引
    i, j, k = get_pool2row_indices(X.shape, kernel_H, kernel_W, stride)
    # 将输入图像X按k,i,j拼接为行
    rows = X.copy()[:, k, i, j]

    return rows.reshape(-1, kernel_H * kernel_W)


# 将行拼接的二维矩阵转换回输入图像X
def row2pool_indices(rows, X_shape, kernel_H, kernel_W, stride):
    batch_size = X_shape[0]
    X = np.zeros(X_shape, dtype=rows.dtype)
    # 获取用于pool2row的索引
    i, j, k = get_pool2row_indices(X_shape, kernel_H, kernel_W, stride)
    rows_reshaped = rows.reshape(batch_size, -1, kernel_H * kernel_W)
    # 将拼接后的列按k,i,j添加到输入图像X中
    np.add.at(X, (slice(None), k, i, j), rows_reshaped)

    # 如果stride小于kernel_H或stride小于kernel_W，则将输入图像X的值设置为1
    if (stride < kernel_H or stride < kernel_W):
        X_ones = np.ones(X.shape)
        rows_ones = X_ones[:, k, i, j]
        X_zeros = np.zeros(X.shape)
        np.add.at(X_zeros, (slice(None), k, i, j), rows_ones)
        return X / X_zeros

    return X


# 将输入图像X转换为输出图像
def pool_fc2output(inputs, batch_size, out_H, out_W):
    output = inputs.copy()
    return output.reshape(batch_size, -1, out_H, out_W)


# 将输出图像转换为输入图像
def pool_output2fc(inputs):
    return inputs.copy().reshape(-1)