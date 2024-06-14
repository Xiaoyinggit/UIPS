# coding:utf8
import tensorflow.compat.v1 as tf
import numpy as np


class NoOff(object):
  def __init__(self, user_count, item_count, user_features,random_seed = 1234,hidden_units=128):
    tf.set_random_seed(random_seed)
    self.user_count = user_count
    self.item_count = item_count
    self.hidden_units = hidden_units
    self.user_features = [tf.convert_to_tensor(user_features[i], dtype=tf.float32) for i in range(len(user_features))]

    self.u = tf.placeholder(tf.int32, [None,])  # [B]
    self.i = tf.placeholder(tf.int32, [None,])  # [B]
    self.y = tf.placeholder(tf.float32, [None,])  # [B]
    self.display = tf.placeholder(tf.float32, [None, ])  # [B]
    self.lr = tf.placeholder(tf.float64, [])
    # evaluate
    self.i10 = tf.placeholder(tf.int32, [None, 10])  # [B,10] 用来做evaluate的

    ############# Pi Network ############
    self.item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units])
    self.item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))
    self.user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
    self.dense_layer = tf.layers.Dense(1, use_bias=False, name='pi_final_mlp')  # 加入的MLP
    # user features
    self.user_emb_layers = [tf.get_variable("Q"+str(i), [5, hidden_units // 4]) for i in range(1,7)]
    self.user_emb_layers.append(tf.get_variable("Q7", [2, hidden_units // 4]))
    self.user_features_final_layer = tf.layers.Dense(self.hidden_units, use_bias=False, name='user_final_emb')

    ################### USER/ITEM EMB###########################
    # self.u_emb = tf.nn.embedding_lookup(self.user_emb_w, self.u)
    self.u_emb = self.getEmb(self.u, self.user_emb_w, self.user_features, self.user_emb_layers,
                             self.user_features_final_layer)  # [B,H]

    ###################### PI ITEM EMB #########################
    self.i_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i)
    self.i_b = tf.gather(self.item_b, self.i)
    batch_size = tf.shape(self.u_emb)[0]
    dis_u_emb_tile = tf.expand_dims(self.u_emb, 1)
    dis_u_emb_tile = tf.tile(dis_u_emb_tile, [1, batch_size, 1])
    dis_i_emb_tile = tf.expand_dims(self.i_emb, 0)
    dis_i_emb_tile = tf.tile(dis_i_emb_tile, [batch_size, 1, 1])
    u_emb_i_emb = tf.multiply(dis_u_emb_tile, dis_i_emb_tile)

    score = self.dense_layer.apply(u_emb_i_emb)
    pi_score = tf.squeeze(score, axis=2) + self.i_b
    onehot_labels = tf.eye(tf.shape(self.u)[0], batch_size)
    loss_per_example = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=pi_score)
    temp_mul = tf.multiply(self.y, loss_per_example)
    self.loss = tf.reduce_mean(temp_mul)

    ############# UPDATE GRADIENT ############
    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

    trainable_params = tf.trainable_variables()
    self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    ############# USED TO EVALUATE############
    self.eval_item_emb = tf.nn.embedding_lookup(self.item_emb_w, self.i10)   # 得到一个 [B,10,H的矩阵]
    self.eval_item_b = tf.gather(self.item_b, self.i10)
    self.u_emb10 = tf.expand_dims(self.u_emb,1)
    self.u_emb10 = tf.tile(self.u_emb10,[1,10,1])  # 因为每个用户只有10个交互
    eval_u_emb_i_emb = tf.multiply(self.u_emb10, self.eval_item_emb)
    self.eval_score = self.dense_layer.apply(eval_u_emb_i_emb)  # [B,10,1]
    self.eval_score = tf.squeeze(self.eval_score, axis=2) + self.eval_item_b  # [B,10]

  def getEmb(self,id,id_emb_layer,features,features_emb_layer,final_layer):
    id_emb = tf.nn.embedding_lookup(id_emb_layer, id)
    features_emb = tf.concat(values=[
      tf.linalg.matmul(tf.gather(features[0], id), features_emb_layer[0]),
      tf.linalg.matmul(tf.gather(features[1], id), features_emb_layer[1]),
      tf.linalg.matmul(tf.gather(features[2], id), features_emb_layer[2]),
      tf.linalg.matmul(tf.gather(features[3], id), features_emb_layer[3]),
      tf.linalg.matmul(tf.gather(features[4], id), features_emb_layer[4]),
      tf.linalg.matmul(tf.gather(features[5], id), features_emb_layer[5]),
      tf.linalg.matmul(tf.gather(features[6], id), features_emb_layer[6])
    ],axis=-1)
    input_emb = tf.concat(values=[id_emb,features_emb],axis=-1)
    input_emb = tf.layers.batch_normalization(inputs=input_emb)
    output_emb = final_layer.apply(input_emb)
    return output_emb




  def train(self, sess, uij, l):
    loss, _= sess.run([self.loss, self.train_op], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.lr: l,
        self.display: uij[3]
        })
    return loss


  def run_evaluate_user(self,sess, u, i10):
      eval_score = sess.run([self.eval_score],feed_dict={
          self.u: u,
          self.i10: i10
      })
      return eval_score

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)