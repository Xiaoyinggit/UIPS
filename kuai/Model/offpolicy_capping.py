# coding:utf8
import tensorflow.compat.v1 as tf
import numpy as np


class Capping(object):
  def __init__(self, user_count, item_count, cate_count, cate_list, capping,random_seed = 1234,hidden_units=128,epsilon=1e-9):
    tf.set_random_seed(random_seed)
    self.user_count = user_count
    self.item_count = item_count
    self.cate_count = cate_count
    self.cate_list = tf.convert_to_tensor(cate_list, dtype=tf.float32)  # [item_count,cate_count]
    self.hidden_units = hidden_units
    self.capping = tf.cast(capping, tf.float32)
    self.epsilon = epsilon

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
    self.beta_sg_u_emb = tf.identity(tf.stop_gradient(self.u_emb))


    ###################### BETA Network ###########################
    self.loss_beta, self.beta_ips_prob =  self.build_beta_network()


    ###################### PI ITEM EMB #########################
    self.i_emb, self.i_b = self.getIemb(self.item_emb_w,self.cate_emb_w,self.item_b)  # [B,H], [B,]
    displayed_index = tf.where(self.display > 0)
    dis_u_emb = tf.squeeze(tf.gather(self.u_emb, displayed_index), axis=1)
    u_emb_i_emb, dis_i_b = self.getTileVector(displayed_index, self.u_emb, self.i_emb, self.i_b)
    score = self.dense_layer.apply(u_emb_i_emb)
    pi_score = tf.squeeze(score,axis=2) + dis_i_b
    pi_prob = tf.keras.layers.Softmax(axis=-1)(pi_score)
    self.pi_prob = tf.diag_part(pi_prob)

    ###########################  Importance weight * Final PI Loss ###############
    self.importance_weight = self.getIPS()  # get IPS
    onehot_labels = tf.eye(tf.shape(dis_u_emb)[0], tf.shape(dis_u_emb)[0])
    loss_per_example = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=pi_score)
    dis_y = tf.squeeze(tf.gather(self.y, displayed_index),axis=1)
    temp_mul = tf.multiply(dis_y, loss_per_example)
    self.loss_pi = tf.reduce_mean(tf.multiply(self.importance_weight, temp_mul))

    ############# UPDATE GRADIENT ############
    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)

    self.loss = self.loss_pi + self.loss_beta
    trainable_params = tf.trainable_variables()
    self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)

    ### joint train
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)
    ### train beta
    gradients_beta = tf.gradients(self.loss_beta, trainable_params)
    clip_gradients_beta, _ = tf.clip_by_global_norm(gradients_beta, 5)
    self.train_op_beta = self.opt.apply_gradients(zip(clip_gradients_beta, trainable_params), global_step=self.global_step)
    ### train pi
    gradients_pi = tf.gradients(self.loss_pi, trainable_params)
    clip_gradients_pi, _ = tf.clip_by_global_norm(gradients_pi, 5)
    self.train_op_pi = self.opt.apply_gradients(zip(clip_gradients_pi, trainable_params), global_step=self.global_step)

    ############# USED TO EVALUATE############
    self.eval_all_i = tf.range(start=0, limit=self.item_count, delta=1)
    self.eval_item_emb, self.eval_item_b = self.getAllItem()
    eval_u_size = tf.shape(self.u_emb)[0]
    eval_i_size = tf.shape(self.eval_item_emb)[0]
    n_u_emb_place = tf.expand_dims(self.u_emb, 1)
    n_u_emb_place = tf.tile(n_u_emb_place, [1, eval_i_size, 1])
    n_iemb_place = tf.expand_dims(self.eval_item_emb, 0)
    n_iemb_place = tf.tile(n_iemb_place, [eval_u_size, 1, 1])
    u_emb_i_emb_place = tf.multiply(n_u_emb_place, n_iemb_place)
    self.eval_score= self.dense_layer.apply(u_emb_i_emb_place)
    self.eval_score = tf.squeeze(self.eval_score,axis=2) + self.eval_item_b



  def build_beta_network(self):
    ############# Beta Network ############
    # beta network
    self.beta_item_emb_w = tf.get_variable("beta_item_emb_w", [self.item_count, self.hidden_units // 2])
    self.beta_item_b = tf.get_variable("beta_item_b", [self.item_count], initializer=tf.constant_initializer(0.0))
    self.beta_cate_emb_w = tf.get_variable("beta_cate_emb_w", [self.cate_count, self.hidden_units // 2])
    self.beta_dense_layer = tf.layers.Dense(1, use_bias=False,name='beta_final_mlp')
    self.beta_i_emb, self.beta_i_b = self.getIemb(self.beta_item_emb_w, self.beta_cate_emb_w, self.beta_item_b)

    ##################### Beta Network Loss #####################
    beta_u_emb_i_emb, beta_i_b = self.getTileVector(None, self.beta_sg_u_emb, self.beta_i_emb,
                                                            self.beta_i_b,useDisplay=False)
    self.logits_beta = self.beta_dense_layer.apply(beta_u_emb_i_emb)
    self.logits_beta = tf.squeeze(self.logits_beta, axis=2) + self.beta_i_b
    onehot_labels = tf.eye(tf.shape(self.u_emb)[0], tf.shape(self.u_emb)[0])
    loss_per_example_beta = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=self.logits_beta)
    self.loss_beta = tf.reduce_mean(tf.multiply(self.display, loss_per_example_beta))

    # get beta prob in IPS
    displayed_index = tf.where(self.display > 0)
    dis_beta_u_emb_i_emb, dis_beta_i_b = self.getTileVector(displayed_index, self.beta_sg_u_emb, self.beta_i_emb, self.beta_i_b)
    self.logits_beta_dis = self.beta_dense_layer.apply(dis_beta_u_emb_i_emb)
    self.logits_beta_dis = tf.squeeze(self.logits_beta_dis, axis=2) + dis_beta_i_b
    beta_prob = tf.keras.layers.Softmax(axis=-1)(self.logits_beta_dis)
    self.beta_ips_prob = tf.diag_part(beta_prob) + self.epsilon

    return self.loss_beta, self.beta_ips_prob



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


  def getTileVector(self, displayed_index,u_emb, i_emb, i_b,useDisplay=True):  # multiply
      if useDisplay:
          u_emb = tf.squeeze(tf.gather(u_emb, displayed_index), axis=1)
          i_emb = tf.squeeze(tf.gather(i_emb, displayed_index),axis=1)
          i_b = tf.squeeze(tf.gather(i_b, displayed_index),axis=1)
      dis_size = tf.shape(u_emb)[0]
      dis_u_emb_tile = tf.expand_dims(u_emb, 1)
      dis_u_emb_tile = tf.tile(dis_u_emb_tile, [1, dis_size, 1])
      dis_i_emb_tile = tf.expand_dims(i_emb, 0)
      dis_i_emb_tile = tf.tile(dis_i_emb_tile, [dis_size, 1, 1])
      u_emb_i_emb = tf.multiply(dis_u_emb_tile, dis_i_emb_tile)
      return u_emb_i_emb, i_b


  def getIPS(self):
      ipsweight = tf.math.divide(tf.stop_gradient(self.pi_prob), tf.stop_gradient(self.beta_ips_prob))
      capping_max = tf.fill(tf.shape(ipsweight), self.capping)
      index_of_capping = tf.greater(ipsweight, capping_max)  # [bool,bool,bool,...]
      ipsweight = tf.where(index_of_capping, capping_max, ipsweight)
      return tf.stop_gradient(ipsweight)

  def train(self, sess, uij, l):
    loss, _ = sess.run([self.loss, self.train_op], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        self.lr: l,
        self.display: uij[5]
        })
    return loss

  def train_pi(self, sess, uij, l):
    loss, _ = sess.run([self.loss_pi, self.train_op_pi], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.hist_i: uij[3],
        self.sl: uij[4],
        self.lr: l,
        self.display: uij[5]
        })
    return loss
  def train_beta(self, sess, uij, l):
    loss, _ = sess.run([self.loss_beta, self.train_op_beta], feed_dict={
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