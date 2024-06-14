# coding:utf8
import tensorflow.compat.v1 as tf
import numpy as np


class NoOff(object):
  def __init__(self, user_count, item_count, cate_count, cate_list, random_seed = 1234,hidden_units=128):
    tf.set_random_seed(random_seed)
    self.user_count = user_count
    self.item_count = item_count
    self.cate_count = cate_count
    self.cate_list = tf.convert_to_tensor(cate_list, dtype=tf.float32)  # [item_count,cate_count]
    self.hidden_units = hidden_units

    self.u = tf.placeholder(tf.int32, [None,])  # [B]
    self.i = tf.placeholder(tf.int32, [None,])  # [B]
    self.y = tf.placeholder(tf.float32, [None,])  # [B]
    self.hist_i = tf.placeholder(tf.int32, [None, None])  # [B, T]
    self.display = tf.placeholder(tf.float32, [None, ])  # [B]
    self.sl = tf.placeholder(tf.int32, [None,])  # [B]
    self.lr = tf.placeholder(tf.float64, [])

    ############# Pi Network ############
    self.item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units // 2])
    self.item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))
    self.cate_emb_w = tf.get_variable("cate_emb_w", [cate_count, hidden_units // 2])
    self.user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
    self.u_dense_layer = tf.layers.Dense(hidden_units, name='u_emb_mlp')
    self.dense_layer = tf.layers.Dense(1, use_bias=False, name='pi_final_mlp')  # 加入的MLP


    ################### USER/ITEM EMB###########################
    self.u_emb = self.getUemb()  # user state embedding


    ###################### PI ITEM EMB #########################
    self.i_emb, self.i_b = self.getIemb(self.item_emb_w,self.cate_emb_w,self.item_b)  # [B,H], [B,]
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
    self.eval_all_i = tf.range(start=0, limit=self.item_count, delta=1)
    self.eval_item_emb, self.eval_item_b = self.getAllItem()
    eval_u_size = tf.shape(self.u_emb)[0]
    eval_i_size = tf.shape(self.eval_item_emb)[0]
    eval_u_emb_tile = tf.expand_dims(self.u_emb, 1)
    eval_u_emb_tile = tf.tile(eval_u_emb_tile, [1, eval_i_size, 1])
    eval_i_emb_tile = tf.expand_dims(self.eval_item_emb, 0)
    eval_i_emb_tile = tf.tile(eval_i_emb_tile, [eval_u_size, 1, 1])
    eval_u_emb_i_emb = tf.multiply(eval_u_emb_tile, eval_i_emb_tile)
    self.eval_score = self.dense_layer.apply(eval_u_emb_i_emb)
    self.eval_score = tf.squeeze(self.eval_score, axis=2) + self.eval_item_b


  def getIemb(self,item_emb_w,cate_emb_w,item_b):
      ic = tf.gather(self.cate_list, self.i)
      i_emb = tf.concat(values=[
          tf.nn.embedding_lookup(item_emb_w, self.i),  # [B,h/2]
          tf.linalg.matmul(ic, cate_emb_w),  # [B,cate_count],[cate_count,h/2]
      ], axis=1)
      i_b = tf.gather(item_b, self.i)
      return i_emb, i_b

  def getAllItem(self):
      i = tf.range(start=0, limit=self.item_count, delta=1)
      all_i_emb = tf.concat(values=[
          tf.nn.embedding_lookup(self.item_emb_w, i),  # [B,h/2]
          tf.linalg.matmul(self.cate_list, self.cate_emb_w)
      ],axis=-1)
      all_i_b = self.item_b
      return all_i_emb, all_i_b


  def getUemb(self):
    u_emb_0 = tf.nn.embedding_lookup(self.user_emb_w, self.u)
    hc = tf.gather(self.cate_list, self.hist_i)  # self.hist_i [B,T,cate_count]
    h_emb = tf.concat([
          tf.nn.embedding_lookup(self.item_emb_w, self.hist_i),  # [B,T,H/2]
          tf.linalg.matmul(hc, self.cate_emb_w),  # [B,T,cate_count] [cate_count,h/2]  # 会出现nan值
    ], axis=2)
    mask = tf.sequence_mask(self.sl, tf.shape(h_emb)[1], dtype=tf.float32)  # [disB, T]
    mask = tf.expand_dims(mask, -1)  # [B, T, 1]
    mask = tf.tile(mask, [1, 1, tf.shape(h_emb)[2]])  # [B, T, H]
    h_emb *= mask  # [B, T, H]
    hist = h_emb
    hist = tf.reduce_sum(hist, 1)
    hist = tf.math.divide_no_nan(hist, tf.cast(tf.tile(tf.expand_dims(self.sl, 1), [1, self.hidden_units]), tf.float32))
    hist = tf.reshape(hist, [-1, self.hidden_units])
    hist = tf.layers.dense(hist, self.hidden_units)
    u_emb = tf.concat([u_emb_0, hist], axis=-1)  # [B,2H]
    u_emb = tf.layers.batch_normalization(inputs=u_emb)
    u_emb = self.u_dense_layer.apply(u_emb)  # [B,H]
    return u_emb



  def train(self, sess, uij, l):
    loss, _= sess.run([self.loss, self.train_op], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        self.lr: l,
        self.display: uij[5]
        })
    return loss


  def run_evaluate_user(self,sess, u, u_hist, sl):
      eval_score = sess.run([self.eval_score],feed_dict={
          self.u: u,
          self.hist_i: u_hist,
          self.sl: sl
      })
      return eval_score

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)