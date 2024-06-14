# coding:utf8
import tensorflow.compat.v1 as tf
import numpy as np

class BetaHat(object):
  def __init__(self, X_count, A_count, hidden_dim=64):
    tf.set_random_seed(1234)
    self.X_count = X_count  # user count
    self.A_count = A_count  # item count 30938
    self.hidden_dim = hidden_dim

    self.X = tf.placeholder(tf.float32, [None, self.hidden_dim])  # [B]
    self.item = tf.placeholder(tf.int32,[None,])
    self.label = tf.placeholder(tf.float32, [None,])  # display
    self.lr = tf.placeholder(tf.float64, [])

    ############# Network ############
    init_A_val = tf.tile(tf.reshape(tf.eye(self.hidden_dim), [1, -1]) , [A_count, 1]) # [A_count, d*d]
    self.A_matrix_2D =  tf.Variable(initial_value=init_A_val, trainable=False, dtype=tf.float32) # uncertainty matrix
    self.initA  = tf.assign(self.A_matrix_2D, init_A_val)
    tf.summary.scalar('A_matrix/A_norm:', tf.norm(self.A_matrix_2D))

    self.item_emb_w = tf.get_variable("item_emb_w", [A_count, self.hidden_dim])
    self.item_b = tf.get_variable("item_b", [1,A_count], initializer=tf.constant_initializer(0.0))



    ################### LOSS###########################
    self.logits = tf.linalg.matmul(self.X, tf.transpose(self.item_emb_w)) + self.item_b
    self.softmax_prob = tf.keras.layers.Softmax(axis=-1)(self.logits)
    onehot_labels =tf.one_hot(self.item, self.A_count)
    # self.uipair_softmax_prob = tf.boolean_mask(self.softmax_prob,onehot_labels)
    loss_per_example = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=self.logits)
    temp_mul = tf.multiply(self.label, loss_per_example)
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

    ############# AUncertainty ############
    grad_beta_diag = tf.math.l2_normalize(self.X, axis=-1) #[B,d]
    grad_beta_diag_a = tf.expand_dims(grad_beta_diag, 1) #[B, 1,d]
    grad_beta_diag_b = tf.expand_dims(grad_beta_diag, 2) #[B, d, 1]
    tmp_A_addi = tf.reshape(tf.matmul(grad_beta_diag_b,grad_beta_diag_a)/self.hidden_dim, [-1, self.hidden_dim* self.hidden_dim]) #[B, d*d]

    item_map = tf.one_hot(self.item, depth=self.A_count) #[B, A]
    A_addi = tf.matmul(tf.transpose(item_map), tmp_A_addi) #[A, d*d]
    self.update_A = tf.assign(self.A_matrix_2D, tf.add(self.A_matrix_2D, A_addi)) 

    # calSUncertainty
    A_matrix_3D = tf.reshape(self.A_matrix_2D, [-1, self.hidden_dim, self.hidden_dim])
    inv_A_matrix = tf.linalg.inv(A_matrix_3D) #[A, d, d] 

    i_A_matrix = tf.gather(inv_A_matrix, self.item)  #[B, d, d]
    uncertainty = tf.squeeze(tf.matmul(tf.matmul(grad_beta_diag_a, i_A_matrix), grad_beta_diag_b)/ self.hidden_dim, axis=2) # [B,1,1]
    print('uncertainty ', uncertainty)
    self.uncertainty_S = tf.math.sqrt(uncertainty) #[B,1]

    # calAUncertainty
    grad_beta_all = tf.tile(tf.expand_dims(grad_beta_diag, 0), [self.A_count, 1,1]) #[A, B,d]
    uncertainty_A = tf.matmul(tf.matmul(grad_beta_all,inv_A_matrix ), tf.transpose(grad_beta_all, [0, 2,1]) )#[A, B, B]

    mask =tf.tile(tf.expand_dims(tf.cast(tf.eye(tf.shape(self.X)[0]), tf.bool), 0), [self.A_count, 1, 1]) #[A, B, B]
    diag_un = tf.reshape(tf.boolean_mask(tensor= uncertainty_A, mask = mask), [self.A_count, -1]) #[A, B]
    self.debug = diag_un

    self.uncertainty_A = tf.sqrt(diag_un/self.hidden_dim) #[A, B]

  
    self.merged = tf.summary.merge_all()

  def train(self, sess, uij, l,summary_writer):
    loss, _, merged_summ, _ = sess.run([self.loss, self.train_op,self.merged, self.update_A], feed_dict={
        self.X: uij[0],
        self.item: uij[1],
        self.label: uij[3],
        self.lr: l
        })
    summary_writer.add_summary(merged_summ, global_step=self.global_step.eval())
    return loss

  def run_evaluate_user(self,sess, x):
      eval_score = sess.run([self.logits],feed_dict={
          self.X: x
      })
      return eval_score

  def run_eval(self, sess, x):
     beta_prob = sess.run([self.softmax_prob], feed_dict={
       self.X: x
     })
     return beta_prob


  
  def getSUncertainty(self,sess,uij):
      un = sess.run([self.uncertainty_S], feed_dict={
          self.X: uij[0],
          self.item: uij[1],
      })
      return un[0]    # [bathsize,1]

  def getAUncertainty(self, sess, x):
      un, debug = sess.run([self.uncertainty_A, self.debug], feed_dict={
          self.X: np.array(x)})
      
      return un[:,0]

  def getAllUncertainty(self, sess, uij):
      un = sess.run([self.uncertainty_A], feed_dict={
          self.X: uij[0]})
      un=np.array(un)
      return un[:,0]

  def save(self, sess, path):
    saver = tf.train.Saver()
    saver.save(sess, save_path=path)

  def restore(self, sess, path):
    saver = tf.train.Saver()
    saver.restore(sess, save_path=path)
