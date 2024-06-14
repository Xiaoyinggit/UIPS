# coding:utf8
import tensorflow.compat.v1 as tf
import numpy as np


class BetaStart(object):
  def __init__(self, X_count, A_count, feature_count, para={'temp':1.0}, hidden_dim=64):
    tf.set_random_seed(1234)
    self.X_count = X_count  # user count
    self.A_count = A_count  # 30938
    self.feature_count = feature_count  # 101938
    self.para=para
    print('para: ', para)

    self.X = tf.placeholder(tf.float32, [None,feature_count])  # [B]
    self.label = tf.placeholder(tf.float32, [None,A_count])  # [B]
    self.lr = tf.placeholder(tf.float64, [])

    ############# Network ############
    self.item_emb_w = tf.get_variable("item_emb_w", [A_count, hidden_dim])
    self.item_b = tf.get_variable("item_b", [1,A_count], initializer=tf.constant_initializer(0.0))
    self.X_layer = tf.layers.Dense(hidden_dim, use_bias=False, name='final_mlp')
    self.X_emb = self.X_layer.apply(self.X) #[B, hidden_dim]

    ################### LOSS###########################
    self.logits = tf.linalg.matmul(self.X_emb, tf.transpose(self.item_emb_w)) + self.item_b
    self.softmax_prob = tf.keras.layers.Softmax(axis=-1)(self.logits/self.para['temp'])
    self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits))

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


  def train(self, sess, uij, l):
    loss, _= sess.run([self.loss, self.train_op], feed_dict={
        self.X: uij[0],
        self.label: uij[1],
        self.lr: l
        })
    return loss

  def run_evaluate_user(self,sess, u):
      eval_score = sess.run([self.logits],feed_dict={
          self.X: u
      })
      return eval_score

  def run_eval(self, sess, uij, l):
     X_emb, beta_prob = sess.run([self.X_emb, self.softmax_prob], feed_dict={
       self.X: uij[0]
     })
     return X_emb, beta_prob


  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)

  # def initFeature(self,sess,feature_list):
  #   sess.run(self.initfeature_list,feed_dict={self.feature_input:feature_list})