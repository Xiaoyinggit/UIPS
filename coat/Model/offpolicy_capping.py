# coding:utf8
import tensorflow.compat.v1 as tf
import numpy as np


class Capping(object):
  def __init__(self, user_count, item_count, user_features,item_features, capping,random_seed = 1234,hidden_units=128,epsilon=1e-9):
    tf.set_random_seed(random_seed)
    self.user_count = user_count
    self.item_count = item_count
    self.hidden_units = hidden_units
    self.capping = tf.cast(capping, tf.float32)
    self.epsilon = epsilon
    self.user_features = [tf.convert_to_tensor(user_features[i], dtype=tf.float32) for i in range(len(user_features))]
    self.item_features = [tf.convert_to_tensor(item_features[i], dtype=tf.float32) for i in range(len(user_features))]

    self.u = tf.placeholder(tf.int32, [None,])  # [B]
    self.i = tf.placeholder(tf.int32, [None,])  # [B]
    self.y = tf.placeholder(tf.float32, [None,])  # [B]
    self.display = tf.placeholder(tf.float32, [None, ])  # [B]
    self.lr = tf.placeholder(tf.float64, [])
    # evaluate
    self.i16 = tf.placeholder(tf.int32, [None, 16])  # [B,16] 用来做evaluate的


    ############# Pi Network ############
    self.item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units])
    self.item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))
    self.user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
    self.dense_layer = tf.layers.Dense(1, use_bias=False, name='pi_final_mlp')  # 加入的MLP
    # user features
    self.user_emb_layers = [tf.get_variable("user_gender", [2, hidden_units // 4]),
                            tf.get_variable("user_age", [6, hidden_units // 4]),
                            tf.get_variable("user_location", [3, hidden_units // 4]),
                            tf.get_variable("user_fashioninterest", [3, hidden_units // 4])]
    self.user_features_final_layer = tf.layers.Dense(self.hidden_units, use_bias=False, name='user_final_emb')
    # item features
    self.item_emb_layers = [tf.get_variable("item_gender", [2, hidden_units // 4]),
                            tf.get_variable("item_jackettype", [16, hidden_units // 4]),
                            tf.get_variable("item_color", [13, hidden_units // 4]),
                            tf.get_variable("item_onfrontpage", [2, hidden_units // 4])]
    self.item_features_final_layer = tf.layers.Dense(self.hidden_units, use_bias=False, name='item_final_emb')

    ################### USER/ITEM EMB###########################
    self.u_emb = self.getEmb(self.u, self.user_emb_w, self.user_features, self.user_emb_layers,
                             self.user_features_final_layer)  # [B,H]
    self.beta_sg_u_emb = tf.identity(tf.stop_gradient(self.u_emb))


    ###################### BETA Network ###########################
    self.loss_beta, self.beta_ips_prob = self.build_beta_network()


    ###################### PI ITEM EMB #########################
    self.i_emb = self.getEmb(self.i, self.item_emb_w, self.item_features, self.item_emb_layers,
                             self.item_features_final_layer)  # [B,H]
    self.i_b = tf.gather(self.item_b, self.i)
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
    dis_y = tf.squeeze(tf.gather(self.y, displayed_index), axis=1)
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
    self.eval_item_emb = self.getEmb(self.i16, self.item_emb_w, self.item_features, self.item_emb_layers, self.item_features_final_layer) #[B,16,H]
    self.eval_item_b = tf.gather(self.item_b, self.i16) # [B,16]
    self.u_emb16 = tf.expand_dims(self.u_emb,1)
    self.u_emb16 = tf.tile(self.u_emb16,[1,16,1])  # 因为每个用户只有16个交互
    eval_u_emb_i_emb = tf.multiply(self.u_emb16, self.eval_item_emb)
    self.eval_score = self.dense_layer.apply(eval_u_emb_i_emb)  # [B,16,1]
    self.eval_score = tf.squeeze(self.eval_score, axis=2) + self.eval_item_b  # [B,16]



  def build_beta_network(self):
    ############# Beta Network ############
    # beta network
    self.beta_item_emb_w = tf.get_variable("beta_item_emb_w", [self.item_count, self.hidden_units])
    self.beta_item_b = tf.get_variable("beta_item_b", [self.item_count], initializer=tf.constant_initializer(0.0))
    self.beta_dense_layer = tf.layers.Dense(1, use_bias=False,name='beta_final_mlp')
    self.beta_item_emb_layers = [tf.get_variable("beta_item_gender", [2, self.hidden_units // 4]),
                            tf.get_variable("beta_item_jackettype", [16, self.hidden_units // 4]),
                            tf.get_variable("beta_item_color", [13, self.hidden_units // 4]),
                            tf.get_variable("beta_item_onfrontpage", [2, self.hidden_units // 4])]
    self.beta_item_features_final_layer = tf.layers.Dense(self.hidden_units, use_bias=False, name='beta_item_final_emb')
    self.beta_i_emb = self.getEmb(self.i, self.beta_item_emb_w, self.item_features, self.beta_item_emb_layers,
                             self.beta_item_features_final_layer)  # [B,H]
    self.beta_i_b = tf.gather(self.beta_item_b, self.i)

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



  def getEmb(self,id,id_emb_layer,features,features_emb_layer,final_layer):
    id_emb = tf.nn.embedding_lookup(id_emb_layer, id)
    features_emb = tf.concat(values=[
      tf.linalg.matmul(tf.gather(features[0], id), features_emb_layer[0]),
      tf.linalg.matmul(tf.gather(features[1], id), features_emb_layer[1]),
      tf.linalg.matmul(tf.gather(features[2], id), features_emb_layer[2]),
      tf.linalg.matmul(tf.gather(features[3], id), features_emb_layer[3])
    ],axis=-1)
    input_emb = tf.concat(values=[id_emb,features_emb],axis=-1)
    input_emb = tf.layers.batch_normalization(inputs=input_emb)
    output_emb = final_layer.apply(input_emb)
    return output_emb

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

  def train(self, sess, uij, l,summary_writer=None):
    loss, _ = sess.run([self.loss, self.train_op], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.lr: l,
        self.display: uij[3]
        })
    return loss

  def train_pi(self, sess, uij, l,summary_writer=None):
    loss, _ = sess.run([self.loss_pi, self.train_op_pi], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.lr: l,
        self.display: uij[3]
        })
    return loss
  def train_beta(self, sess, uij, l,summary_writer=None):
    loss, _ = sess.run([self.loss_beta, self.train_op_beta], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.lr: l,
        self.display: uij[3]
        })
    return loss


  def run_evaluate_user(self,sess, u,i16):
      eval_score = sess.run([self.eval_score],feed_dict={
          self.u: u,
          self.i16:i16
      })
      return eval_score

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)