import tensorflow as tf
import numpy as np


class SMM():
    def __init__(self, s, w, l2_reg, model_type, embeddings, num_features, d0=300, di=150, d_att1 = 8, d_att2 = 16, w_att1 = 6, w_att2 = 4, num_classes=2, num_layers=2, corr_w = 0.1):

        self.x1 = tf.placeholder(tf.int32, shape=[None, s], name="x1")
        self.x2 = tf.placeholder(tf.int32, shape=[None, s], name="x2")
        self.y = tf.placeholder(tf.int32, shape=[None], name="y")
        self.features = tf.placeholder(tf.float32, shape=[None, num_features], name="features")
        l2_reg_lambda = l2_reg
        last_output_layer_size = di * 4 + d_att2 * (s/4/3/2) * (s/4/3/2)

        self.corr_w = corr_w
        #embeddings = tf.Variable(tf.random_uniform([vocab_size, 300], -1.0, 1.0))
        self.E = tf.Variable(embeddings, trainable=True, dtype=tf.float32)

        emb1_ori = tf.nn.embedding_lookup(self.E, self.x1)
        emb2_ori = tf.nn.embedding_lookup(self.E, self.x2)

        emb1 = tf.transpose(emb1_ori, [0, 2, 1], name="emb1_trans")
        emb2 = tf.transpose(emb2_ori, [0, 2, 1], name="emb2_trans")

        # zero padding to inputs for wide convolution
        def pad_for_wide_conv(x):
            return tf.pad(x, np.array([[0, 0], [0, 0], [w - 1, w - 1], [0, 0]]), "CONSTANT", name="pad_wide_conv")

        def cos_sim(v1, v2):
            norm1 = tf.sqrt(tf.reduce_sum(tf.square(v1), axis=1))
            norm2 = tf.sqrt(tf.reduce_sum(tf.square(v2), axis=1))
            dot_products = tf.reduce_sum(v1 * v2, axis=1, name="cos_sim")

            return dot_products / (norm1 * norm2)

        def euclidean_score(v1, v2):
            euclidean = tf.sqrt(tf.reduce_sum(tf.square(v1 - v2), axis=1))
            return 1 / (1 + euclidean)

        def make_attention_mat(x1, x2):
            # x1, x2 = [batch, height, width, 1] = [batch, d, s, 1]
            # x2 => [batch, height, 1, width]
            # [batch, width, wdith] = [batch, s, s]
            #euclidean = tf.sqrt(tf.reduce_sum(tf.square(x1 - tf.matrix_transpose(x2)), axis=1))
            dot = tf.reduce_sum(tf.matmul(x1,tf.matrix_transpose(x2)), axis=1, name="att_sim")
            return dot
        def layer_attention(variable_scope,layer_dot,x1,x2):
            with tf.variable_scope(variable_scope):
                # return => [batch,width,width] <= [batch,width_x1,width_x2]
                #layer_dot = make_attention_mat(x1,x2)
                # x1-attention:
                # [batch,width_x1,width_x2] => [batch,width_x1]
                x1att_w = tf.reduce_sum(layer_dot, 2)
                print('{}  shape: {}'.format(x1att_w.name,x1att_w.shape))
                x1att_alphas = tf.nn.softmax(x1att_w, name='x1att_alphas')  
                # op: transpose => [bacth,1,width,1]
                x1att_alphas_new = tf.transpose(tf.expand_dims(tf.expand_dims(x1att_alphas,-1),-1),[0,2,1,3])
                x1att = tf.multiply(x1,x1att_alphas_new)
                print('{}  shape: {}'.format(x1att.name,x1att.shape))
                # x2-attention:
                # [batch,width_x1,width_x2] => [batch,width_x2]
                x2att_w = tf.reduce_sum(layer_dot,1)
                print('{}  shape: {}'.format(x2att_w.name,x2att_w.shape))
                x2att_alphas = tf.nn.softmax(x2att_w, name='x2att_alphas')  
                # op: transpose => [bacth,1,width,1]
                x2att_alphas_new = tf.transpose(tf.expand_dims(tf.expand_dims(x2att_alphas,-1),-1),[0,2,1,3])
                x2att = tf.multiply(x2,x2att_alphas_new)
                print('{}  shape: {}'.format(x2att.name,x2att.shape))
                return x1att,x2att,x1att_alphas,x2att_alphas
                        
        def convolution(name_scope,variable_scope, x, d,reuse=False):
            # Convolution layer for BCNN
            with tf.name_scope(name_scope + "-conv"):
                with tf.variable_scope(variable_scope) as scope:
                    conv = tf.contrib.layers.conv2d(
                        inputs=x,
                        num_outputs=di,
                        kernel_size=(d, w),
                        stride=1,
                        padding="VALID",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=reuse,
                        trainable=True,
                        scope=scope)
                    # Weight: [filter_height, filter_width, in_channels, out_channels]
                    # output: [batch, 1, input_width+filter_Width-1, out_channels] == [batch, 1, s+w-1, di]

                    # [batch, di, s+w-1, 1]
                    conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                    return conv_trans
                
        def convolution1(name_scope, variable_scope, x, kernel_vector, d_output):
            with tf.name_scope(name_scope + "-conv1"):
                with tf.variable_scope(variable_scope) as scope:
                    conv = tf.contrib.layers.conv2d(
                        inputs=x,
                        num_outputs=d_output,
                        kernel_size=kernel_vector,
                        stride=1,
                        padding="SAME",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=False,
                        trainable=True,
                        scope=scope
                    )
                    # Input: [batch, s, s, 1]
                    # Weight: [filter_height, filter_width, in_channels, out_channels]
                    # output: [batch, s-d+1, s-d+1, d_output]

                    # [batch, di, s+w-1, 1]
                    #conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                    # [batch, s-d+1, s-d+1, d_output]
                    conv_trans = conv
                    return conv_trans

        def convolution2(name_scope, variable_scope, x, kernel_vector, d_output):
            with tf.name_scope(name_scope + "-conv2"):
                with tf.variable_scope(variable_scope) as scope:
                    conv = tf.contrib.layers.conv2d(
                        inputs=x,
                        num_outputs=d_output,
                        kernel_size=kernel_vector,
                        stride=3,
                        padding="VALID",
                        activation_fn=tf.nn.tanh,
                        weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                        weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                        biases_initializer=tf.constant_initializer(1e-04),
                        reuse=False,
                        trainable=True,
                        scope=scope
                    )
                    # Input: [batch, s, s, 1]
                    # Weight: [filter_height, filter_width, in_channels, out_channels]
                    # output: [batch, s-d+1, s-d+1, d_output]

                    # [batch, di, s+w-1, 1]
                    #conv_trans = tf.transpose(conv, [0, 3, 2, 1], name="conv_trans")
                    # [batch, s-d+1, s-d+1, d_output]
                    conv_trans = conv
                    return conv_trans

        def w_pool(variable_scope, x, attention):
            # Window Pooling layer for BCNN (if necessary, this is used as the first layer)
            # x: [batch, di, s+w-1, 1]
            # attention: [batch, s+w-1]
            with tf.variable_scope(variable_scope + "-w_pool"):
                if model_type == "ABCNN2" or model_type == "ABCNN3":
                    pools = []
                    # [batch, s+w-1] => [batch, 1, s+w-1, 1]
                    attention = tf.transpose(tf.expand_dims(tf.expand_dims(attention, -1), -1), [0, 2, 1, 3])

                    for i in range(s):
                        # [batch, di, w, 1], [batch, 1, w, 1] => [batch, di, 1, 1]
                        pools.append(tf.reduce_sum(x[:, :, i:i + w, :] * attention[:, :, i:i + w, :],
                                                   axis=2,
                                                   keep_dims=True))

                    # [batch, di, s, 1]
                    w_ap = tf.concat(pools, axis=2, name="w_ap")
                else:
                    w_ap = tf.layers.average_pooling2d(
                        inputs=x,
                        # (pool_height, pool_width)
                        pool_size=(1, w),
                        strides=1,
                        padding="VALID",
                        name="w_ap"
                    )
                    # [batch, di, s, 1]

                return w_ap
        '''
        def con_att_pool(variable_scope,attention):
            with tf.variable_scope(variable_scope):
                c_ap = tf.layers.max_pooling2d(
                    # [batch,di,s,1]
                    inputs = attention,
                    pool_size = (1,w),
                    strides = 1,
                    padding="VALID"
                    name = "c_ap"
                )
                #[batch,di,s,1]
                return c_ap
        '''
        def all_att_pool1(variable_scope, x, pool_vector):
            with tf.variable_scope(variable_scope + "-all_pool"):
                all_ap = tf.layers.max_pooling2d(
                    inputs=x,
                    # (pool_height, pool_width)
                    pool_size=pool_vector,
                    strides=4,
                    padding="VALID",
                    name="all_ap"
                )
                # [batch, di, 1, 1]

                return all_ap

        def all_att_pool2(variable_scope, x, pool_vector, d_hidden):
            with tf.variable_scope(variable_scope + "-all_pool"):
                all_ap = tf.layers.max_pooling2d(
                    inputs=x,
                    # (pool_height, pool_width)
                    pool_size=pool_vector,
                    strides=2,
                    padding="VALID",
                    name="all_ap"
                )
                # [batch, di, 1, 1]

                # [batch, di]
                all_ap_reshaped = tf.reshape(all_ap, [-1, d_hidden*(s/4/3/2) * (s/4/3/2)])
                #all_ap_reshaped = tf.squeeze(all_ap, [2, 3])

                return all_ap_reshaped

        def all_pool(variable_scope, x):
            # All Pooling Layer for BCNN
            with tf.variable_scope(variable_scope + "-all_pool"):
                if variable_scope.startswith("input"):
                    pool_width = s
                    d = d0
                    all_ap = tf.layers.average_pooling2d(
                        inputs=x,
                        # (pool_height, pool_width)
                        pool_size=(1, pool_width),
                        strides=1,
                        padding="VALID",
                        name="all_ap"
                    )
                else:
                    pool_width = s + w - 1
                    d = di
                    all_ap = tf.layers.max_pooling2d(
                        inputs=x,
                        # (pool_height, pool_width)
                        pool_size=(1, pool_width),
                        strides=1,
                        padding="VALID",
                        name="all_ap"
                    )
                # [batch, di, 1, 1]

                # [batch, di]
                all_ap_reshaped = tf.reshape(all_ap, [-1, d])
                #all_ap_reshaped = tf.squeeze(all_ap, [2, 3])

                return all_ap_reshaped
        

        def CNN_layer(variable_scope, x1, x2, d):
            # x1, x2 = [batch, d, s, 1]
            with tf.variable_scope(variable_scope):
                with tf.name_scope("att_mat"):
                    # [batch, s, s]
                    att_mat = make_attention_mat(x1, x2)

                left_attention1,right_attention1,a_1,a_2 = layer_attention("layer_att1",att_mat,x1,x2)
                att_mat_expanded = tf.expand_dims(att_mat, -1)
                
                
                left_attconv = convolution(name_scope="left",variable_scope='leftconv', x=pad_for_wide_conv(left_attention1), d=d)
                right_attconv = convolution(name_scope="right",variable_scope='rightconv', x=pad_for_wide_conv(right_attention1), d=d)
                left_conv = convolution(name_scope="left_",variable_scope='leftconv_', x=pad_for_wide_conv(x1), d=d)
                right_conv = convolution(name_scope="right_",variable_scope='rightconv_', x=pad_for_wide_conv(x2), d=d)
                att_mat2 = make_attention_mat(left_conv,right_conv)
                left_attention2,right_attention2,a_11,a_22 = layer_attention("layer_att2",att_mat2,left_conv,right_conv)
                

                #left_wp = w_pool(variable_scope="left", x=left_conv, attention=left_attention)
                left_ap = all_pool(variable_scope="left", x=left_attention2)
                left_att_ap = all_pool(variable_scope="left_att_pool",x=left_attconv)
                #print('left all pooling shape {}'.format(left_ap.shape))
                #right_wp = w_pool(variable_scope="right", x=right_conv, attention=right_attention)
                right_ap = all_pool(variable_scope="right", x=right_attention2)
                right_att_ap = all_pool(variable_scope="right_att_pool",x=right_attconv)
                #print('right all pooling shape {}'.format(right_ap.shape))
                print('attention mat shape{}'.format(att_mat_expanded.shape))
                
                
                att_conv = convolution1(name_scope="att", variable_scope="att", x=att_mat_expanded, kernel_vector=(w_att1,w_att1), d_output=d_att1) 
                #padding='SAME',shape=[batch,ceil(s/stride),ceil(s/stride),d_att1]
                print(att_conv.shape)
                att_ap = all_att_pool1(variable_scope="att", x=att_conv, pool_vector=(4,4)) 
                #padding='VALID',shape=[batch,ceil((s-kernel+1)/stride),ceil((s-kernel+1)/stride),d_att1]
                print(att_ap.shape)
                att_conv2 = convolution2(name_scope="att2", variable_scope="att2", x=att_ap, kernel_vector=(w_att2,w_att2), d_output=d_att2)
                print(att_conv2.shape)
                att_ap2 = all_att_pool2(variable_scope="att2", x=att_conv2, pool_vector=(2,2), d_hidden=d_att2)
                print('attention conv shape {}'.format(att_ap2.shape))
                return left_att_ap, left_ap, right_att_ap, right_ap, att_ap2,att_mat,a_1,a_2,a_11,a_22

        x1_expanded = tf.expand_dims(emb1, -1)
        x2_expanded = tf.expand_dims(emb2, -1)

        src_LI_1, src_LO_1, src_RI_1, src_RO_1, src_att_ap,src_att_mat,_,_,_,_ = CNN_layer(variable_scope="src_CNN-1", x1=x1_expanded, x2=x2_expanded, d=d0)
        #src_sims = [cos_sim(LO_0, RO_0)]
        # concat and diff
        src_diff = tf.subtract(tf.concat([src_LI_1,src_LO_1],axis=1), tf.concat([src_RI_1,src_RO_1],axis=1))
        print('src_subtract shape {}'.format(src_diff.shape))
        src_mul = tf.multiply(tf.concat([src_LI_1,src_LO_1],axis=1), tf.concat([src_RI_1,src_RO_1],axis=1))
        print('src_mul shape {}'.format(src_mul.shape))
        LI_1, LO_1, RI_1, RO_1, att_ap,att_mat,_,_,_,_ = CNN_layer(variable_scope="CNN-1", x1=x1_expanded, x2=x2_expanded, d=d0)
        #sims = [cos_sim(LO_0, RO_0)]
        diff = tf.subtract(tf.concat([LI_1, LO_1],axis=1), tf.concat([RI_1,RO_1],axis=1))
        print('share_subtract shape {}'.format(diff.shape))
        mul = tf.multiply(tf.concat([LI_1, LO_1],axis=1), tf.concat([RI_1,RO_1],axis=1))
        print('share_mul shape {}'.format(mul.shape))

        tgt_LI_1, tgt_LO_1, tgt_RI_1, tgt_RO_1, tgt_att_ap,tgt_att_mat,tgt_a1,tgt_a2,tgt_a11,tgt_a22 = CNN_layer(variable_scope="tgt_CNN-1", x1=x1_expanded, x2=x2_expanded, d=d0)
        #tgt_sims = [cos_sim(LO_0, RO_0)]
        tgt_diff = tf.subtract(tf.concat([tgt_LI_1, tgt_LO_1],axis=1), tf.concat([tgt_RI_1, tgt_RO_1],axis=1))
        tgt_mul = tf.multiply(tf.concat([tgt_LI_1, tgt_LO_1],axis=1), tf.concat([tgt_RI_1, tgt_RO_1],axis=1))
        #sims = [cos_sim(LO_0, RO_0)]
        self.tgt_att_mat = tgt_att_mat
        self.tgt_a1 = tgt_a1
        self.tgt_a2 = tgt_a2
        self.tgt_a11 = tgt_a11
        self.tgt_a22 = tgt_a22

        self.src_output_features = tf.concat([src_att_ap, src_LI_1,src_LO_1, src_RI_1,src_RO_1, src_diff, src_mul], axis=1, name="src_output_features")
        print(self.src_output_features.shape)
        self.shared_output_features = tf.concat([att_ap, LI_1,LO_1, RI_1, RO_1, diff, mul], axis=1, name="shared_output_features")
        print(self.shared_output_features.shape)
        self.tgt_output_features = tf.concat([tgt_att_ap, tgt_LI_1,tgt_LO_1, tgt_RI_1,tgt_RO_1, tgt_diff, tgt_mul], axis=1, name="tgt_output_features")
        print(self.tgt_output_features.shape)
        with tf.variable_scope("shared-src-output-layer"):
            self.shared_src_estimation = tf.contrib.layers.fully_connected(
                inputs=self.shared_output_features,
                num_outputs=num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        with tf.variable_scope("shared-tgt-output-layer"):
            self.shared_tgt_estimation = tf.contrib.layers.fully_connected(
                inputs=self.shared_output_features,
                num_outputs=num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        with tf.variable_scope("src-output-layer"):
            self.src_estimation = tf.contrib.layers.fully_connected(
                inputs=self.src_output_features,
                num_outputs=num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        with tf.variable_scope("tgt-output-layer"):
            self.tgt_estimation = tf.contrib.layers.fully_connected(
                inputs=self.tgt_output_features,
                num_outputs=num_classes,
                activation_fn=None,
                weights_initializer=tf.contrib.layers.xavier_initializer(),
                weights_regularizer=tf.contrib.layers.l2_regularizer(scale=l2_reg),
                biases_initializer=tf.constant_initializer(1e-04),
                scope="FC"
            )

        self.src_prediction = tf.contrib.layers.softmax(self.src_estimation + self.shared_src_estimation)[:, :]

        self.src_pred_cost = tf.add(
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.src_estimation + self.shared_src_estimation, labels=self.y)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            name="src_cost")

        self.tgt_prediction = tf.contrib.layers.softmax(self.tgt_estimation + self.shared_tgt_estimation)[:, :]

        self.tgt_pred_cost = tf.add(
            tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.tgt_estimation + self.shared_tgt_estimation, labels=self.y)),
            tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)),
            name="tgt_cost")

        with tf.variable_scope("src-output-layer", reuse = True):
            src_W_s = tf.get_variable("FC/weights")
        with tf.variable_scope("shared-src-output-layer", reuse = True):
            src_W_c = tf.get_variable("FC/weights")
        with tf.variable_scope("shared-tgt-output-layer", reuse = True):
            tgt_W_c = tf.get_variable("FC/weights")
        with tf.variable_scope("tgt-output-layer", reuse = True):
            tgt_W_t = tf.get_variable("FC/weights")

        src_W_s_reshaped = tf.reshape(src_W_s, [-1, self.src_output_features.shape[1] * num_classes])
        src_W_c_reshaped = tf.reshape(src_W_c, [-1, self.shared_output_features.shape[1] * num_classes])
        tgt_W_t_reshaped = tf.reshape(tgt_W_t, [-1, self.tgt_output_features.shape[1] * num_classes])
        tgt_W_c_reshaped = tf.reshape(tgt_W_c, [-1, self.shared_output_features.shape[1] * num_classes])

        # weight matrix: 4*H, H=2+d_att2
        self.weights = tf.concat([src_W_s_reshaped, src_W_c_reshaped, tgt_W_t_reshaped, tgt_W_c_reshaped], 0)

        # to model the correlation between weights
        self.site_corr = tf.placeholder('float', [4, 4])
        trans_w = tf.transpose(self.weights)
        self.corr2 = tf.matmul(trans_w, tf.matmul(self.site_corr, self.weights))
        # self.corr2 = tf.matmul(tf.matmul(trans_w, self.site_corr), self.weights['out'])
        print('Corr2', self.corr2)
        self.corr_trace = tf.trace(self.corr2)
        print('Corr_trace', self.corr_trace)

        # for updating site_corr
        self.site_corr_trans = tf.matmul(self.weights, trans_w)

        # 0.001 is the lambda
        self.src_cost = self.src_pred_cost + tf.reduce_sum(l2_reg_lambda*self.corr_trace*self.corr_w)

        # 0.001 is the lambda
        self.tgt_cost = self.tgt_pred_cost + tf.reduce_sum(l2_reg_lambda*self.corr_trace*self.corr_w)

        #self.corr_cost = tf.reduce_sum(l2_reg_lambda*self.corr_trace*self.corr_w)

        tf.summary.scalar("cost", self.src_cost)
        tf.summary.scalar("cost", self.tgt_cost)
        self.merged = tf.summary.merge_all()

        print("=" * 50)
        print("List of Variables:")
        for v in tf.trainable_variables():
            print(v.name)
        print("=" * 50)
