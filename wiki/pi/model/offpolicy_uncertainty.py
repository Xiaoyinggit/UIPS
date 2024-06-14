# coding:utf8
import tensorflow.compat.v1 as tf
import numpy as np

class UIPS(object):
  def __init__(self, X_count, A_count,para, random_seed = 1234,hidden_dim=64):
    tf.set_random_seed(random_seed)
    self.X_count = X_count  # user count
    self.A_count = A_count  # item count 30938
    self.hidden_dim = hidden_dim
    self.lambda_ = tf.cast(para['lambda_'], tf.float32)
    self.eta_ = tf.cast(para['eta_'], tf.float32)
    self.para = para
    print('[pi_UIPS] para: ', self.para)

    self.X = tf.placeholder(tf.float32, [None, self.hidden_dim])  # [B]
    self.item = tf.placeholder(tf.int32,[None,])
    self.label = tf.placeholder(tf.float32, [None,])  # reward
    self.display = tf.placeholder(tf.float32, [None,])  # [B]
    self.lr = tf.placeholder(tf.float64, [])
    self.beta_prob = tf.placeholder(tf.float32, [None,self.A_count])
    self.ori_beta_uncertainty = tf.placeholder(tf.float32, [None,])

    ############# Network ############
    self.item_emb_w = tf.get_variable("item_emb_w", [A_count, self.hidden_dim])
    self.item_b = tf.get_variable("item_b", [1,A_count], initializer=tf.constant_initializer(0.0))
    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
  



    ################### LOSS###########################
    self.logits = tf.linalg.matmul(self.X, tf.transpose(self.item_emb_w)) + self.item_b
    self.softmax_prob = tf.keras.layers.Softmax(axis=-1)(self.logits)
    onehot_labels =tf.one_hot(self.item, self.A_count)
    loss_per_example = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=self.logits)
    temp_mul = tf.multiply(self.label, loss_per_example)
    self.importance_weight = self.getIPS()
    print('importance_weight: ', self.importance_weight )
    print('temp_mul', temp_mul)
    print('ips*tmp_nul:', tf.multiply(self.importance_weight, temp_mul))
    self.loss = tf.reduce_mean(tf.multiply(self.importance_weight, temp_mul))



    ############# UPDATE GRADIENT ############
    trainable_params = tf.trainable_variables()
    self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)


    self.merged = tf.summary.merge_all()

  def getIPS(self):
    small_v = 0.000001
    index = tf.one_hot(self.item, self.A_count)
    pi_prob_sg = tf.stop_gradient(self.softmax_prob)
    pi_prob_sg = tf.boolean_mask(pi_prob_sg, index)
    beta_prob = tf.boolean_mask(self.beta_prob,index)
    ipsweight = tf.math.divide(pi_prob_sg, beta_prob+small_v) #[B,1]
    print('ipsweight: ', ipsweight)
    tf.summary.scalar('ipsweight/ipsweight_mean', tf.reduce_mean(ipsweight))
    tf.summary.scalar('ipsweight/ipsweight_mean', tf.reduce_mean(ipsweight))
    tf.summary.scalar('ipsweight/ipsweight_min', tf.reduce_min(ipsweight))
    tf.summary.scalar('ipsweight/ipsweight_max', tf.reduce_max(ipsweight))

    ##########################UNCERTAINTY REWEIGHT######################
    diag_beta_dis_uty = tf.exp(-self.para['gamma'] * self.ori_beta_uncertainty)

    print('diag_beta_dis_uty:', diag_beta_dis_uty)
    tf.summary.scalar('beta_dis_uty/min_beta_dis_uty', tf.reduce_min(self.ori_beta_uncertainty))
    tf.summary.scalar('beta_dis_uty/max_beta_dis_uty', tf.reduce_max(self.ori_beta_uncertainty))
    tf.summary.scalar('beta_dis_uty/mean_beta_dis_uty', tf.reduce_mean(self.ori_beta_uncertainty))
    tf.summary.scalar('beta_dis_uty/min_exp_beta_dis_uty', tf.reduce_min(diag_beta_dis_uty))
    tf.summary.scalar('beta_dis_uty/max_exp_beta_dis_uty', tf.reduce_max(diag_beta_dis_uty))
    tf.summary.scalar('beta_dis_uty/mean_exp_beta_dis_uty', tf.reduce_mean(diag_beta_dis_uty))

    pi_beta_U = tf.math.divide(tf.square(pi_prob_sg),
                               tf.multiply(tf.square(beta_prob), diag_beta_dis_uty) + small_v)
    print('pi_beta_U: ', pi_beta_U)
    tf.summary.scalar('pi_beta_U/min_pi_beta_U', tf.reduce_min(pi_beta_U))
    tf.summary.scalar('pi_beta_U/max_pi_beta_U', tf.reduce_max(pi_beta_U))
    tf.summary.scalar('pi_beta_U/mean_pi_beta_U', tf.reduce_mean(pi_beta_U))

    phi_sa_1 = self.lambda_ / (self.lambda_ / self.eta_ * diag_beta_dis_uty + self.eta_ * pi_beta_U + small_v)
    phi_sa_2 = 2* self.para['eta_2'] / (tf.exp(-self.para['gamma'] * self.ori_beta_uncertainty) + tf.exp(self.para['gamma'] * self.ori_beta_uncertainty))
    phi_sa = tf.math.minimum (phi_sa_1, phi_sa_2)
    print('phi_sa: ', phi_sa, 'phi_sa_1: ', phi_sa_1, 'phi_sa_2:', phi_sa_2)
    tf.summary.scalar('phi_sa/ori_phi_sa_mean', tf.reduce_mean(phi_sa))
    tf.summary.scalar('phi_sa/ori_phi_sa_min', tf.reduce_min(phi_sa))
    tf.summary.scalar('phi_sa/ori_phi_sa_max', tf.reduce_max(phi_sa))

    if self.para['normalize_phi_sa'] > 0:
      phi_sa = phi_sa / (tf.reduce_sum(phi_sa) + 0.000001)
    tf.summary.scalar('phi_sa/phi_sa_mean', tf.reduce_mean(phi_sa))
    tf.summary.scalar('phi_sa/phi_sa_min', tf.reduce_min(phi_sa))
    tf.summary.scalar('phi_sa/phi_sa_max', tf.reduce_max(phi_sa))

    #################################GET Final IPS #######################
    ips_final = tf.multiply(ipsweight, phi_sa)
    print('ips_final', ips_final)
    tf.summary.scalar('ips_final/ips_final_mean', tf.reduce_mean(ips_final))
    tf.summary.scalar('ips_final/ips_final_min', tf.reduce_min(ips_final))
    tf.summary.scalar('ips_final/ips_final_max', tf.reduce_max(ips_final))

    ips_final_cond = tf.equal(ips_final, tf.reduce_max(ips_final))
    tf.summary.scalar('debug/ips_final_max_phi_sa: ',
                      tf.reduce_mean(tf.boolean_mask(tensor=phi_sa, mask=ips_final_cond)))
    tf.summary.scalar('debug/ips_final_max_ipsweight: ',
                      tf.reduce_mean(tf.boolean_mask(tensor=ipsweight, mask=ips_final_cond)))
    tf.summary.scalar('debug/ips_final_max_diag_beta_dis_uty: ',
                      tf.reduce_mean(tf.boolean_mask(tensor=diag_beta_dis_uty, mask=ips_final_cond)))
    tf.summary.scalar('debug/ips_final_max_pi_beta_U: ',
                      tf.reduce_mean(tf.boolean_mask(tensor=pi_beta_U, mask=ips_final_cond)))
    tf.summary.scalar('debug/ips_final_max_ips_final: ',
                      tf.reduce_mean(tf.boolean_mask(tensor=tf.multiply(ipsweight, phi_sa), mask=ips_final_cond)))

    return tf.stop_gradient(ips_final)

  def train(self, sess, uij, l, beta_prob,beta_uncertainty,summary_writer):
    loss, _, merged_summ = sess.run([self.loss, self.train_op,self.merged], feed_dict={
        self.X: uij[0],
        self.item: uij[1],
        self.label: uij[2],
        self.display:uij[3],
        self.lr: l,
        self.beta_prob: beta_prob,
        self.ori_beta_uncertainty: beta_uncertainty
        })
    summary_writer.add_summary(merged_summ, global_step=self.global_step.eval())
    return loss

  def run_evaluate_user(self,sess, x):
      eval_score = sess.run([self.logits],feed_dict={
          self.X: x
      })
      return eval_score

  def run_eval(self, sess, x):
     pi_prob = sess.run([self.softmax_prob], feed_dict={
       self.X: x
     })
     return pi_prob


  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)
