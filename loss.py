from torch.nn.modules.loss import CrossEntropyLoss


def loss_calu(predict, target, config):
    loss_fn = CrossEntropyLoss()
    batch_img, batch_attr, batch_obj, batch_target = target
    batch_attr = batch_attr.cuda()
    batch_obj = batch_obj.cuda()
    batch_target = batch_target.cuda()
    logits_att, logits_obj = predict[0], predict[1]
    loss = loss_fn(logits_obj, batch_obj) + loss_fn(logits_att, batch_attr)
    return loss