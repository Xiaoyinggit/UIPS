# coding:utf8
import tensorflow.compat.v1 as tf
import numpy as np

class Pi_capping(object):
  def __init__(self, X_count, A_count,capping, random_seed = 1234,hidden_dim=64,epsilon=1e-9):
    tf.set_random_seed(random_seed)
    self.X_count = X_count  # user count
    self.A_count = A_count  # item count 30938
    self.hidden_dim = hidden_dim
    self.capping = tf.cast(capping, tf.float32)
    self.epsilon = epsilon

    self.X = tf.placeholder(tf.float32, [None, self.hidden_dim])  # [B]
    self.item = tf.placeholder(tf.int32,[None,])
    self.label = tf.placeholder(tf.float32, [None,])  # reward
    self.display = tf.placeholder(tf.float32, [None,])  # [B]
    self.lr = tf.placeholder(tf.float64, [])
    self.beta_prob = tf.placeholder(tf.float32, [None,self.A_count])
    self.beta_gt_prob = tf.placeholder(tf.float32, [None,])
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
    self.loss = tf.reduce_mean(tf.multiply(self.importance_weight, temp_mul))


    ############# UPDATE GRADIENT ############
    trainable_params = tf.trainable_variables()
    self.opt = tf.train.AdamOptimizer(learning_rate=self.lr)
    gradients = tf.gradients(self.loss, trainable_params)
    clip_gradients, _ = tf.clip_by_global_norm(gradients, 5)
    self.train_op = self.opt.apply_gradients(zip(clip_gradients, trainable_params), global_step=self.global_step)

    self.merged = tf.summary.merge_all()



  def getIPS(self):
    index = tf.one_hot(self.item, self.A_count)
    pi_prob_sg = tf.stop_gradient(self.softmax_prob)
    pi_prob_sg = tf.boolean_mask(pi_prob_sg, index)
    beta_prob = tf.boolean_mask(self.beta_prob,index)
    # summary 
    tf.summary.scalar('beta_prob/estimated:', tf.reduce_mean(beta_prob))
    pos_index = tf.greater(self.label, 0)
    tf.summary.scalar('beta_prob/ pos_num:', tf.reduce_sum(tf.cast(pos_index, tf.float32)))
    print('pos_index:', pos_index)
    print('beta_prob: ', beta_prob)
    print('beta_prob_pos: ', tf.boolean_mask(tensor=beta_prob,mask=pos_index))
    tf.summary.scalar('beta_prob/estimated_pos: ', tf.reduce_mean(tf.boolean_mask(tensor=beta_prob,mask=pos_index)))
    tf.summary.scalar('beta_prob/estimated_neg: ', tf.reduce_mean(tf.boolean_mask(tensor=beta_prob,mask=tf.math.logical_not(pos_index))))

    # gt_summary
    beta_gt_prob = tf.reshape(self.beta_gt_prob, tf.shape(pi_prob_sg))
    print('beta_gt: ', beta_gt_prob)
    tf.summary.scalar('beta_prob/gt:', tf.reduce_mean(beta_gt_prob))
    tf.summary.scalar('beta_prob/pos: ', tf.reduce_mean(tf.boolean_mask(tensor=beta_gt_prob,mask=pos_index)))
    tf.summary.scalar('beta_prob/neg: ', tf.reduce_mean(tf.boolean_mask(tensor=beta_gt_prob,mask=tf.math.logical_not(pos_index))))



    ipsweight = tf.math.divide(pi_prob_sg, beta_prob+ self.epsilon) #[B,1]
    tf.summary.scalar('ipsweight/estimated: ', tf.reduce_mean(ipsweight))
    tf.summary.scalar('ipsweight/estimated_pos: ', tf.reduce_mean(tf.boolean_mask(tensor=ipsweight,mask=pos_index)))
    tf.summary.scalar('ipsweight/estimated_neg: ', tf.reduce_mean(tf.boolean_mask(tensor=ipsweight,mask=tf.math.logical_not(pos_index))))

    ipsweight_gt = tf.math.divide(pi_prob_sg, beta_gt_prob+self.epsilon)
    tf.summary.scalar('ipsweight/gt: ', tf.reduce_mean(ipsweight_gt))
    tf.summary.scalar('ipsweight/gt_pos: ', tf.reduce_mean(tf.boolean_mask(tensor=ipsweight_gt,mask=pos_index)))
    tf.summary.scalar('ipsweight/gt_neg: ', tf.reduce_mean(tf.boolean_mask(tensor=ipsweight_gt,mask=tf.math.logical_not(pos_index))))



    capping_max = tf.fill(tf.shape(ipsweight), self.capping)
    index_of_capping = tf.greater(ipsweight, capping_max)
    ips_final = tf.where(index_of_capping, capping_max, ipsweight)
    return tf.stop_gradient(ips_final)

  def train(self, sess, uij, l, beta_prob,beta_uncertainty,summary_writer):
    loss, _, merged_summ = sess.run([self.loss, self.train_op, self.merged], feed_dict={
        self.X: uij[0],
        self.item: uij[1],
        self.label: uij[2],
        self.display:uij[3],
        self.lr: l,
        self.beta_prob: beta_prob,
        self.ori_beta_uncertainty: beta_uncertainty,
        self.beta_gt_prob: uij[4]
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
