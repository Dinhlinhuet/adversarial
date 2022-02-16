import torch

def semantic_attk(generator, end_model, adversary, loss_func, image, mask, mask_tg, mask_tg1, targeted):
    # print('img', image.shape, mask.shape, mask_tg.shape)
    # print('dtype1', image.dtype)
    fake_feature = generator.encoding(image, mask)
    # print('img',image.shape, mask_tg.shape, mask_tg1.shape)
    fake_feature1 = generator.encoding(image, mask_tg1)
    # fake_image1 = generator(image, mask)
    # fake_image1 = generator(image, mask_tg)
    # fake_image1 = generator.decoding(
    #     (fake_feature * 0.2 + fake_feature1 * 0.8))
    # fake_image1 = generator.decoding(
    #     (fake_feature * 0.9 + fake_feature1 * 0.1))
    edit_final, adv_loss = adversary(G_dec=generator.decoding,
                                     emb1=fake_feature,
                                     emb2=fake_feature1,
                                     model=end_model,
                                     loss_func=loss_func,
                                     target_label=mask_tg,
                                     targeted=targeted)
    # edit_final, adv_loss, tv_loss = adversary(G_dec=generator.decoding,
    #                                  emb1=fake_feature,
    #                                  emb2=fake_feature1,
    #                                  model=end_model,
    #                                  loss_func=loss_func,
    #                                  target_label=mask_tg,
    #                                  targeted=targeted)
    return edit_final
    # return fake_image1

def semantic_attk1(generator, end_model, adversary, loss_func, image, mask, mask_tg, mask_tg1, targeted):
    # print('img', image.shape, mask.shape, mask_tg.shape)
    # print('dtype1', image.dtype)
    fake_feature = generator.encoding(image, mask)
    # print('img',image.shape, mask_tg.shape, mask_tg1.shape)
    fake_feature1 = generator.encoding(image, mask_tg1)
    # fake_image1 = generator(image, mask)
    # fake_image1 = generator(image, mask_tg)
    fake_image1 = generator.decoding(
        (fake_feature * 0.2 + fake_feature1 * 0.8))
    # fake_image1 = generator.decoding(
    #     (fake_feature * 0.8 + fake_feature1 * 0.2))
    # edit_final, adv_loss = adversary(G_dec=generator.decoding,
    #                                  emb1=fake_feature,
    #                                  emb2=fake_feature1,
    #                                  model=end_model,
    #                                  loss_func=loss_func,
    #                                  target_label=mask_tg,
    #                                  targeted=targeted)
    # edit_final, adv_loss, tv_loss = adversary(G_dec=generator.decoding,
    #                                  emb1=fake_feature,
    #                                  emb2=fake_feature1,
    #                                  model=end_model,
    #                                  loss_func=loss_func,
    #                                  target_label=mask_tg,
    #                                  targeted=targeted)
    # return edit_final
    return fake_image1