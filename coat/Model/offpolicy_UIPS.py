# coding:utf8
import tensorflow.compat.v1 as tf
import numpy as np


class UIPS(object):
  def __init__(self, user_count, item_count, user_features,item_features,  para,  random_seed = 1234,hidden_units=128):
    tf.set_random_seed(random_seed)
    self.user_count = user_count
    self.item_count = item_count
    self.hidden_units = hidden_units
    self.user_features = [tf.convert_to_tensor(user_features[i], dtype=tf.float32) for i in range(len(user_features))]
    self.item_features = [tf.convert_to_tensor(item_features[i], dtype=tf.float32) for i in range(len(user_features))]

    self.lambda_ = tf.cast(para['lambda_'], tf.float32)
    self.eta_ = tf.cast(para['eta_'], tf.float32)
    self.para = para
    print('[Model_UIPS] para: ', para)

    ##################PLACEHOLDER#######################
    self.A = tf.Variable(initial_value=np.identity(hidden_units), trainable=False, dtype=tf.float32)
    self.u = tf.placeholder(tf.int32, [None,])  # [B]
    self.i = tf.placeholder(tf.int32, [None,])  # [B]
    self.y = tf.placeholder(tf.float32, [None,])  # [B]
    self.display = tf.placeholder(tf.float32, [None, ])  # [B]
    self.lr = tf.placeholder(tf.float64, [])
    # Step variable
    self.global_step = tf.Variable(0, trainable=False, name='global_step')
    self.global_epoch_step = tf.Variable(0, trainable=False, name='global_epoch_step')
    self.global_epoch_step_op = tf.assign(self.global_epoch_step, self.global_epoch_step + 1)
    self.initA = tf.assign(self.A, np.identity(hidden_units))



    #########################PI Network ##############################
    self.item_emb_w = tf.get_variable("item_emb_w", [item_count, hidden_units])
    self.item_b = tf.get_variable("item_b", [item_count], initializer=tf.constant_initializer(0.0))
    self.user_emb_w = tf.get_variable("user_emb_w", [user_count, hidden_units])
    self.dense_layer = tf.layers.Dense(1, use_bias=False, name='pi_final_mlp') 
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


    self.u_emb = self.getEmb(self.u, self.user_emb_w, self.user_features, self.user_emb_layers,
                             self.user_features_final_layer)  # [B,H]
    self.i_emb = self.getEmb(self.i, self.item_emb_w, self.item_features, self.item_emb_layers,
                             self.item_features_final_layer)  # [B,H]
    self.i_b = tf.gather(self.item_b, self.i)

    ###################### BETA Network ###########################
    self.beta_sg_u_emb = tf.identity(tf.stop_gradient(self.u_emb))
    self.loss_beta, self.beta_ips_prob, self.beta_uncertainty, self.beta_updateA =  self.build_beta_network()
   
    ####################### PI Network ##########################
    # pi network only consider samples with display=1
    displayed_index = tf.where(self.display>0)
    dis_u_emb = tf.squeeze(tf.gather(self.u_emb, displayed_index),axis=1) #[disB, d]
    dis_i_emb = tf.squeeze(tf.gather(self.i_emb, displayed_index),axis=1)
    dis_i_b = tf.squeeze(tf.gather(self.i_b, displayed_index),axis=1)# [disB, ]
    dis_size =  tf.shape(dis_u_emb)[0]
    dis_n_u_emb = tf.tile(tf.expand_dims(dis_u_emb, 1), [1, dis_size, 1]) #[disB, disB, d]
    dis_n_i_emb = tf.tile(tf.expand_dims(dis_i_emb, 0), [dis_size, 1, 1])
    u_emb_i_emb = tf.multiply(dis_n_u_emb, dis_n_i_emb) #[disB, disB, d]
    pi_score = tf.squeeze(self.dense_layer.apply(u_emb_i_emb), axis=2) + dis_i_b
    self.pi_prob = tf.diag_part(tf.keras.layers.Softmax(axis=-1)(pi_score))


    ############################ LOSS #############################
    self.importance_weight = self.getIPS(displayed_index)
    onehot_labels = tf.eye(tf.shape(dis_u_emb)[0], tf.shape(dis_u_emb)[0])
    loss_per_example = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=pi_score)
    dis_y = tf.squeeze(tf.gather(self.y, displayed_index),axis=1)
    temp_mul = tf.multiply(dis_y-self.para['lambdaDiff'], loss_per_example)
    self.loss_pi = tf.reduce_mean(tf.multiply(self.importance_weight, temp_mul)) 
    tf.summary.scalar('loss/loss_per_example: ', tf.reduce_mean(temp_mul))
    tf.summary.scalar('loss/loss_pi: ', tf.reduce_mean(self.loss_pi))
    

   ############# UPDATE GRADIENT ############

    self.loss = self.loss_beta + self.loss_pi
    tf.summary.scalar('loss/loss: ', tf.reduce_mean(self.loss))


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


    self.merged =  tf.summary.merge_all()
    ############# USED TO EVALUATE############
    self.i16 = tf.placeholder(tf.int32, [None, 16])  # [B,10] 用来做evaluate的
    self.eval_item_emb = self.getEmb(self.i16, self.item_emb_w, self.item_features, self.item_emb_layers, self.item_features_final_layer) #[B,16,H]
    self.eval_item_b = tf.gather(self.item_b, self.i16) # [B,16]
    self.u_emb16 = tf.expand_dims(self.u_emb,1)
    self.u_emb16 = tf.tile(self.u_emb16,[1,16,1])  # 因为每个用户只有16个交互
    eval_u_emb_i_emb = tf.multiply(self.u_emb16, self.eval_item_emb)
    self.eval_score = self.dense_layer.apply(eval_u_emb_i_emb)  # [B,16,1]
    self.eval_score = tf.squeeze(self.eval_score, axis=2) + self.eval_item_b  # [B,16]

 
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
    # get loss
    batch_size = tf.shape(self.beta_sg_u_emb)[0]
    n_u_emb = tf.expand_dims(self.beta_sg_u_emb, 1)
    n_u_emb = tf.tile(n_u_emb, [1, batch_size, 1])
    beta_n_iemb = tf.expand_dims(self.beta_i_emb, 0)
    beta_n_iemb = tf.tile(beta_n_iemb, [batch_size, 1, 1])
    beta_u_emb_i_emb = tf.multiply(tf.stop_gradient(n_u_emb), beta_n_iemb)
    self.logits_beta = self.beta_dense_layer.apply(beta_u_emb_i_emb)
    self.logits_beta = tf.squeeze(self.logits_beta,axis=2) + self.beta_i_b 

    onehot_labels = tf.eye(tf.shape(self.u_emb)[0], tf.shape(self.u_emb)[0])
    loss_per_example_beta = tf.nn.softmax_cross_entropy_with_logits_v2(labels=onehot_labels, logits=self.logits_beta)
    self.loss_beta = tf.reduce_mean(tf.multiply(self.display, loss_per_example_beta))
    tf.summary.scalar('loss/loss_beta: ', tf.reduce_mean(self.loss_beta))

    # get beta prob in IPS
    displayed_index = tf.where(self.display > 0)
    dis_beta_u_emb = tf.squeeze(tf.gather(self.beta_sg_u_emb, displayed_index),axis=1)
    dis_beta_i_emb = tf.squeeze(tf.gather(self.beta_i_emb, displayed_index),axis=1)
    dis_beta_i_b = tf.squeeze(tf.gather(self.beta_i_b, displayed_index),axis=1)# [disB, ]
    dis_size =  tf.shape(dis_beta_u_emb)[0]
    dis_beta_n_u_emb = tf.tile(tf.expand_dims(dis_beta_u_emb, 1), [1, dis_size, 1]) #[disB, disB, d]
    dis_beta_n_i_emb = tf.tile(tf.expand_dims(dis_beta_i_emb, 0), [dis_size, 1, 1])
    dis_beta_u_emb_i_emb = tf.multiply(dis_beta_n_u_emb, dis_beta_n_i_emb) #[disB, disB, d] 

    self.logits_beta_dis = self.beta_dense_layer.apply(dis_beta_u_emb_i_emb)
    self.logits_beta_dis = tf.squeeze(self.logits_beta_dis,axis=2) + dis_beta_i_b
    beta_prob = tf.keras.layers.Softmax(axis=-1)(self.logits_beta_dis)
    epsilon = 0.0000001
    self.beta_ips_prob = tf.stop_gradient(tf.diag_part(beta_prob)) + epsilon
    ################### Uncertainty #############################
    self.beta_uncertainty, self.beta_updateA = self.calUncertainty(dis_beta_u_emb_i_emb)

    return self.loss_beta, self.beta_ips_prob, self.beta_uncertainty, self.beta_updateA

  def calUncertainty(self,dis_beta_u_emb_i_emb):
      dis_size = tf.shape(dis_beta_u_emb_i_emb)[0]
      grad_beta_2D = dis_beta_u_emb_i_emb  # [disB,disB,128]
      temp_G = tf.reshape(grad_beta_2D, [dis_size * dis_size, self.hidden_units])
      ind = tf.tile(tf.reshape(tf.eye(dis_size, dis_size), [dis_size * dis_size, -1]), [1,self.hidden_units])
      grad_beta_diag = tf.boolean_mask(tensor=temp_G, mask=ind)  # [disB,128]
      grad_beta_diag_raw = tf.reshape(grad_beta_diag,[-1,self.hidden_units]) #[disB, 128]--diagnoal veral
      grad_beta_diag = tf.math.l2_normalize(grad_beta_diag_raw, axis=-1)
      invA = tf.linalg.inv(self.A)
      tf.summary.scalar('A/invA_norm: ', tf.norm(invA))
      tf.summary.scalar('A/A_norm:', tf.norm(self.A))

      uncertainty = tf.diag_part(tf.matmul(tf.matmul(grad_beta_diag, invA), tf.transpose(grad_beta_diag))/self.hidden_units)
      uncertainty = tf.math.sqrt(uncertainty) #[disB,1]
      additive = tf.matmul(tf.transpose(grad_beta_diag), grad_beta_diag)/self.hidden_units
      tf.summary.scalar('A/additive:', tf.norm(additive))



      self.beta_updateA_op = tf.assign(self.A, tf.add(self.A, additive))

      return tf.stop_gradient(uncertainty), self.beta_updateA_op 
  
    
  def getIPS(self,displayed_index):
      pi_prob_sg = tf.stop_gradient(self.pi_prob)
      ipsweight = tf.math.divide(pi_prob_sg,self.beta_ips_prob)
      tf.summary.scalar('ipsweight/ipsweight_mean', tf.reduce_mean(ipsweight))
      tf.summary.scalar('ipsweight/ipsweight_min', tf.reduce_min(ipsweight))
      tf.summary.scalar('ipsweight/ipsweight_max', tf.reduce_max(ipsweight))

      ##########################UNCERTAINTY REWEIGHT######################
      diag_beta_dis_uty =tf.exp(-self.para['gamma']*self.beta_uncertainty)
      tf.summary.scalar('beta_dis_uty/min_beta_dis_uty', tf.reduce_min(self.beta_uncertainty))
      tf.summary.scalar('beta_dis_uty/max_beta_dis_uty', tf.reduce_max(self.beta_uncertainty))
      tf.summary.scalar('beta_dis_uty/mean_beta_dis_uty', tf.reduce_mean(self.beta_uncertainty))
      tf.summary.scalar('beta_dis_uty/min_exp_beta_dis_uty', tf.reduce_min(diag_beta_dis_uty))
      tf.summary.scalar('beta_dis_uty/max_exp_beta_dis_uty', tf.reduce_max(diag_beta_dis_uty))
      tf.summary.scalar('beta_dis_uty/mean_exp_beta_dis_uty', tf.reduce_mean(diag_beta_dis_uty))

      
      pi_beta_U = tf.math.divide(tf.square(pi_prob_sg), tf.multiply(tf.square(self.beta_ips_prob), diag_beta_dis_uty)+0.0000001)
      tf.summary.scalar('pi_beta_U/min_pi_beta_U', tf.reduce_min(pi_beta_U))
      tf.summary.scalar('pi_beta_U/max_pi_beta_U', tf.reduce_max(pi_beta_U))
      tf.summary.scalar('pi_beta_U/mean_pi_beta_U', tf.reduce_mean(pi_beta_U))


      phi_sa_1 = self.lambda_ / (self.lambda_ / self.eta_ * diag_beta_dis_uty + self.eta_ * pi_beta_U + 0.000001)
      phi_sa_2 = 2* self.para['eta_2'] / (tf.exp(-self.para['gamma'] * self.beta_uncertainty) + tf.exp(self.para['gamma'] * self.beta_uncertainty))
      phi_sa = tf.math.minimum (phi_sa_1, phi_sa_2)
      print('phi_sa: ', phi_sa, 'phi_sa_1: ', phi_sa_1, 'phi_sa_2:', phi_sa_2)

      tf.summary.scalar('phi_sa/ori_phi_sa_mean', tf.reduce_mean(phi_sa))
      tf.summary.scalar('phi_sa/ori_phi_sa_min', tf.reduce_min(phi_sa))
      tf.summary.scalar('phi_sa/ori_phi_sa_max', tf.reduce_max(phi_sa))

      if self.para['normalize_phi_sa'] >0 :
         phi_sa = phi_sa/ ( tf.reduce_sum(phi_sa) +0.000001)
      tf.summary.scalar('phi_sa/phi_sa_mean', tf.reduce_mean(phi_sa))
      tf.summary.scalar('phi_sa/phi_sa_min', tf.reduce_min(phi_sa))
      tf.summary.scalar('phi_sa/phi_sa_max', tf.reduce_max(phi_sa))

      #################################GET Final IPS #######################
      is_capping = tf.cast(tf.equal(self.global_epoch_step, 0), dtype=tf.float32) * self.para['cappingFirstEpoch'] 
      is_capping_vec = tf.cast(is_capping* tf.ones_like(ipsweight), tf.bool)
      ips_final_capping = tf.where(tf.greater(ipsweight, self.para['cappingThre']), self.para['cappingThre']*tf.ones_like(ipsweight), ipsweight)
      ips_final_un = tf.multiply(ipsweight, phi_sa) 

      ips_final = tf.where(is_capping_vec, ips_final_capping, ips_final_un)

       
      #print_op = tf.print('ips_final_un, ', tf.shape(ips_final_un), 'ips_final_capping: ', tf.shape(ips_final_capping), 'ips_final: ', tf.shape(ips_final))
      # with tf.control_dependencies([print_op]):
      #   ips_final = tf.identity(ips_final)

      tf.summary.scalar('ips_final/is_capping_vec', tf.reduce_sum(tf.cast(is_capping_vec, tf.float32)))
      tf.summary.scalar('ips_final/ips_final_capping', tf.reduce_mean(ips_final_capping))
      tf.summary.scalar('ips_final/ips_final_un', tf.reduce_mean(ips_final_un))
      tf.summary.scalar('ips_final/ips_final_mean', tf.reduce_mean(ips_final))
      tf.summary.scalar('ips_final/ips_final_min', tf.reduce_min(ips_final))
      tf.summary.scalar('ips_final/ips_final_max', tf.reduce_max(ips_final))

      ips_final_cond = tf.equal(ips_final, tf.reduce_max(ips_final))
      tf.summary.scalar('debug/ips_final_max_phi_sa: ', tf.reduce_mean(tf.boolean_mask(tensor=phi_sa, mask=ips_final_cond)))
      tf.summary.scalar('debug/ips_final_max_ipsweight: ', tf.reduce_mean(tf.boolean_mask(tensor=ipsweight, mask=ips_final_cond)))
      tf.summary.scalar('debug/ips_final_max_diag_beta_dis_uty: ', tf.reduce_mean(tf.boolean_mask(tensor=diag_beta_dis_uty, mask=ips_final_cond)))
      tf.summary.scalar('debug/ips_final_max_pi_beta_U: ', tf.reduce_mean(tf.boolean_mask(tensor=pi_beta_U, mask=ips_final_cond)))
      tf.summary.scalar('debug/ips_final_max_ips_final: ', tf.reduce_mean(tf.boolean_mask(tensor= tf.multiply(ipsweight, phi_sa) , mask=ips_final_cond)))

      return tf.stop_gradient(ips_final)





  def train_pi(self, sess, uij, l, summary_writer):
    loss, _, merged_summ = sess.run([self.loss_pi, self.train_op_pi,self.merged], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.lr: l,
        self.display: uij[3]
        })
    summary_writer.add_summary(merged_summ, global_step=self.global_step.eval())
    return loss

  def train_beta(self, sess, uij, l, summary_writer):
    loss, _, _,  merged_summ= sess.run([self.loss_beta, self.train_op_beta, self.beta_updateA,  self.merged], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.lr: l,
        self.display: uij[3]
        })
    summary_writer.add_summary(merged_summ, global_step=self.global_step.eval())
    return loss

  def train(self, sess, uij, l, summary_writer):
    loss, _, _, merged_summ = sess.run([self.loss, self.train_op, self.beta_updateA, self.merged], feed_dict={
        self.u: uij[0],
        self.i: uij[1],
        self.y: uij[2],
        self.lr: l,
        self.display: uij[3],
        })
    summary_writer.add_summary(merged_summ, global_step=self.global_step.eval())
    return loss
  
  def run_evaluate_user(self,sess, u, i16):
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

