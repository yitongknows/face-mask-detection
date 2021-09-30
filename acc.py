def filterLowProbBoxes(boxes,thre = 0.2):
    keep = []
    for i in range(len(boxes)):
        if boxes[i][4] < 0.2:
            continue
        keep.append(i)
    return keep


def getBoxes(pred, gridN = 13):
    pred = pred.squeeze()
    boxes = np.ones((13*13, 7))
    c = 0
    gridS = 1
    for i in range(13):
        for j in range(13):
            if pred[i][j][2].detach().numpy() > pred[i][j][3].detach().numpy():
                boxes[c,0:4] = pred[i][j][4:8].detach().numpy()
                boxes[c,4] = pred[i][j][2].detach().numpy()
            else:
                boxes[c , 0:4] = pred[i][j][8:12].detach().numpy()
                boxes[c ,4] = pred[i][j][3].detach().numpy()

            xc, yc, w, h = boxes[c, 0], boxes[c, 1], boxes[c, 2], boxes[c, 3]

            xc = (i * gridS + xc * gridS) / gridN
            yc = (j * gridS + yc * gridS) / gridN


            x1, y1, x2, y2 = xc - w/2, yc - h/2, xc + w/2, yc + h/2
            boxes[c,0:4] = np.array([x1, y1, x2, y2])
            boxes[c, 5:7] = pred[i][j][0:2].detach().numpy()
            c += 1
    return boxes

def calculateAccForSingleImg (labels, predictions, thre = 0.2, iou_thre = 0.4):
    labels = getBoxes(labels)
    predictions = getBoxes(predictions)
    labels = labels[filterLowProbBoxes(labels,thre = thre)]
    predictions = predictions[filterLowProbBoxes(predictions,thre = thre)]
    index = torchvision.ops.nms( torch.from_numpy(predictions[:,0:4]).float(), torch.from_numpy(predictions[:,4]).float(), 0.01 ).numpy()
    predictions = predictions[index]
    tacc = 0
    fp = 0
    fn = 0
    tp = 0
    tn = 0
    taken = []

    iou_matrix = torchvision.ops.box_iou(torch.from_numpy(predictions[:,0:4]).float(), torch.from_numpy(labels[:,0:4]).float())

    for j in range(len(labels)):
        if labels[j,5:7].argmax() == 0:
            tp += 1
        else:
            tn += 1

    for i in range(len(predictions)):
        max_iou = -1
        mindex = 0
        for j in range(len(labels)):
            if j in taken:
                continue
            iou =  iou_matrix[i][j].item()
            if iou > max_iou:
                max_iou = iou
                mindex = j

        if max_iou > iou_thre:
            taken.append(mindex)
            if predictions[i,5:7].argmax() == labels[mindex,5:7].argmax():
                tacc += 1
            elif predictions[i,5:7].argmax() == 0:
                fp += 1
            elif predictions[i,5:7].argmax() == 1:
                fn += 1

    detecAcc = len(taken) / len(labels)
    taccr = tacc / len(labels)

    return {
        "detectionAcc" : detecAcc,
        "generalAcc" : taccr,
        "fpRate" : fp / len(taken),
        "fnRate" : fn / len(taken),
        "totalFacesInLabels" : len(labels),
        "totalFacesSuccessfullyPred" : len(taken),
        "totalFacesSuccessfullyPredandClassified" : tacc,
        "fpNum" : fp,
        "fnNum" : fn,
        "tpNum" :tp,
        "tnNum" : tn,
        "wrongDetectionNum" : len(predictions) - len(taken),

    }

