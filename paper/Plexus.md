

## .
## .
Latest updates: hps://dl.acm.org/doi/10.1145/3721146.3721938
## .
## .
## RESEARCH-ARTICLE
Practical Federated Learning without a Server
AKASH BALASAHEB DHASADE, EPFL, Lausanne, Switzerland
## .
ANNE MARIE KERMARREC, EPFL, Lausanne, Switzerland
## .
ERICK LAVOIE, University of Basel, Basel, BS, Switzerland
## .
JOHAN A POUWELSE, Del University of Technology, Del, Zuid-Holland, Netherlands
## .
RISHI SHARMA, EPFL, Lausanne, Switzerland
## .
MARTIJN DE VOS, EPFL, Lausanne, Switzerland
## .
## .
## .
Open Access Support provided by:
## .
University of Basel
## .
## EPFL
## .
Del University of Technology
## .
PDF Download
## 3721146.3721938.pdf
## 22 December 2025
## Total Citations: 0
## Total Downloads: 273
## .
## .
## Published: 30 March 2025
## .
## .
Citation in BibTeX format
## .
## .
EuroMLSys '25: 5th Workshop on
Machine Learning and Systems
## March 30 - April 3, 2025
## Rotterdam, Netherlands
## .
## .
## Conference Sponsors:
## SIGOPS
EuroMLSys '25: Proceedings of the 5th Workshop on Machine Learning and Systems (March 2025)
hps://doi.org/10.1145/3721146.3721938
## ISBN: 9798400715389
## .

Practical Federated Learning without a Server
AkashDhasade
## EPFL
## Lausanne, Switzerland
Anne-Marie Kermarrec
## EPFL
## Lausanne, Switzerland
## Erick Lavoie
University of Basel
## Basel, Switzerland
## Johan Pouwelse
Delft University of Technology
## Delft, The Netherlands
## Rishi Sharma
## EPFL
## Lausanne, Switzerland
Martijn de Vos
## EPFL
## Lausanne, Switzerland
## Abstract
Federated Learning (FL) enables end-user devices to collabo-
ratively train ML models without sharing raw data, thereby
preserving data privacy. In FL, a central parameter server
coordinates the learning process by iteratively aggregating
the trained models received from clients. Yet, deploying a
central server is not always feasible due to hardware un-
availability, infrastructure constraints, or operational costs.
We present Plexus, a fully decentralized FL system for large
networks that operates without the drawbacks originating
from having a central server. Plexus distributes the respon-
sibilities of model aggregation and sampling among partic-
ipating nodes while avoiding network-wide coordination.
We evaluate Plexus using realistic traces for compute speed,
pairwise latency and network capacity. Our experiments on
three common learning tasks and with up to1000nodes
empirically show that Plexus reduces time-to-accuracy by
1.4-1.6×, communication volume by 15.8-292×and training
resources needed for convergence by 30.5-77.9×compared
to conventional decentralized learning algorithms.
CCS Concepts:•Computing methodologies→Distributed
algorithms;•Computer systems organization→Peer-
to-peer architectures.
Keywords:Federated Learning, Decentralized Learning, De-
centralized Peer Sampling
ACM Reference Format:
Akash Dhasade, Anne-Marie Kermarrec, Erick Lavoie, Johan Pouwelse,
Rishi Sharma, and Martijn de Vos. 2025. Practical Federated Learn-
ing without a Server. InThe 5th Workshop on Machine Learning and
Permission to make digital or hard copies of all or part of this work for
personal or classroom use is granted without fee provided that copies
are not made or distributed for prot or commercial advantage and that
copies bear this notice and the full citation on the rst page. Copyrights
for components of this work owned by others than the author(s) must
be honored. Abstracting with credit is permitted. To copy otherwise, or
republish, to post on servers or to redistribute to lists, requires prior specic
permission and/or a fee. Request permissions from permissions@acm.org.
EuroMLSys ’25, March 30-April 3 2025, Rotterdam, Netherlands
©2025 Copyright held by the owner/author(s). Publication rights licensed
to ACM.
## ACM ISBN 979-8-4007-1538-9/2025/03
hps://doi.org/10.1145/3721146.3721938
Systems (EuroMLSys ’25), March 30-April 3 2025, Rotterdam, Nether-
lands.ACM, New York, NY, USA, 11 pages.hps://doi.org/10.1145/
## 3721146.3721938
## 1  Introduction
Federated learning (FL) enables devices (referred to asnodes
in this work) to collaboratively train a global machine learn-
ing (ML) model without sharing their private training data.
In a singleFLtraining round, a central server rst selects
asample,i.e., a random subset of online nodes, that train a
model in parallel [41]. Nodes then send their updated model
to the server, which aggregates incoming models into a sin-
gle model.FLis widely used today in various applications,
including next-word prediction on keyboards [13,14,20],
speech recognition [53], human activity recognition [55],
and healthcare [11, 38, 49, 52].
However, deploying and maintaining a centralFLserver
can be challenging for various reasons [51,65]. Infrastructure
constraints in remote or underdeveloped areas may make it
dicult to set up a reliable central server. Additionally, regu-
latory and privacy concerns might restrict centralized model
aggregation, particularly in sensitive domains like health-
care and nance [44]. Furthermore, the reliance on a central
server can introduce a single point of failure, stalling training
progression if the server becomes unavailable [18,28,68].
Despite these challenges, centralized model aggregation used
byFLoers an advantage over existing decentralized learn-
ing (DL) approaches that rely on neighborhood-based aggre-
gation sinceFLresults in higher model accuracy and faster
model convergence. We thus ask ourselves the question:Can
we perform FL without a server?
We answer armatively and presentPlexus, a fully de-
centralized approach forFLtraining. The core ofPlexuslies
in itsdecentralized peer samplerthat enables nodes to inde-
pendently determine a subset of other nodes, or asample,
that is in charge of the training process for a given round.
The sample changes every round, therefore evenly balancing
the training load among nodes and providing nodes with an
equal opportunity to contribute to model training. Follow-
ing local model training, nodes in a sample select a single
aggregatorfrom the same sample that aggregates all trained
models generated in each round. This aggregator then sends
the aggregated model to nodes in the next sample, initiating
the next round of training.
## 1

EuroMLSys ’25, March 30-April 3 2025, Roerdam, NetherlandsDhasade et al.
## 20
## 40
## 60
## 02505007501000
## Communication Rounds
## Test Accuracy
## FLDL (D−PSGD)
Figure 1.The evolution of test accuracy ofFLandDL(D-
PSGD) on the CIFAR-10 dataset in a1000-node network.FL
converges quicker and to higher accuracy than DL.
We evaluatePlexususing real-world mobile phone traces
of pairwise latencies, bandwidth capacities, and computation
speeds [32]. Our evaluation covers three common learning
tasks in varying network sizes, up to1000nodes. We com-
pare the performance ofPlexusagainst standardFLand two
baselineDLalgorithms: decentralized parallel stochastic gra-
dient descent (D-PSGD) [35] and gossip learning (GL) [21].
Our experimental results show thatPlexus, compared to the
best performingDLbaseline, reduces time-to-accuracy by
1.4-1.6×, communication volume by 15.8-292×, and training
resources consumed by 30.5-77.9×. Furthermore, we demon-
strate thatPlexuscompetes with the performance ofFLin
real-world settings.
This work makes the following two contributions:
1.We designPlexus, a practical and decentralizedFL
system (Section 3).Plexusincorporates a decentral-
ized peer sampler to select a small subset of nodes
that train the model each round, signicantly reducing
training resources required to converge compared to
existingDLapproaches. Our system operates without
any centralized or network-wide coordination.
## 2.
We implementPlexusand conduct an extensive evalu-
ation ofPlexususing real-world mobile phone traces
at scale, comparing it with prominent baselineDLalgo-
rithms and standardFLwith a server (Section 4). Our
results demonstrate thatPlexus, compared toDLbase-
lines, signicantly enhances performance across three
common learning tasks regarding time-to-accuracy,
communication volume, and resource requirements,
while providing similar accuracy to FL.
## 2  Motivation
Federated learning (FL)is a collaborativeMLalgorithm
where a central parameter server orchestrates the train-
ing ofMLmodels on client devices. The eectiveness and
practicality ofFLhas been well-documented in various set-
tings [11,13,38,49,52,53,55]. However, its heavy reliance
on a central server to coordinate training throughclient
samplingand sample-widegradient aggregationmakes
## FL
impractical in many real-world scenarios. In particular,FL
training can last for days, and the central server needs to re-
main continuously available throughout the training process.
Furthermore, the central parameter server inFLdemands a
high-bandwidth network connection to communicate with
multiple clients simultaneously. These high availability and
network bandwidth requirements result in signicant opera-
tional and infrastructure costs for the FL infrastructure.
DLand Residual Variance.DLemerges as a promis-
ing alternative to
FL[5].DLapproaches likeD-PSGD[35],
Epidemic learning (EL) [16],GL[46], and derivatives [6,7]
eliminate the need for the central server by introducing peer-
to-peer communication and model aggregation. However,
theseDLapproaches often do not attain the same accura-
cies asFL. This is because nodes inDLaggregate models
amongst neighborhoods,i.e., local aggregation [
## 35]. Local
aggregation leavesresidual variancebetween local models,
which biases gradient computations and slows down model
convergence compared to performing a global aggregation
before starting a training round [4, 29].
To further illustrate this eect, we chart in Figure 1 the test
accuracy forFLandD-PSGDas the training progresses, on
the CIFAR-10 dataset under an Independent and Identically
Distributed (IID) data distribution in a network with1000
nodes. We implementD-PSGDwith the state-of-the-art one-
peer exponential graph topology (OP-Exp.) [64]. With this
topology, each node receives and sends exactly one model
every round. A peer is connected toĢĥĝ(Ĥ)neighbors (Ĥis
the total network size) and cycles through them round-robin.
All other learning parameters follow our experimental setup
described in Section 4.1. Figure 1 shows thatFL’s aggregation
is benecial for convergence and reaches higher accuracy
than DL by nullifying residual variance across nodes in the
network.
3  Design ofPlexus
We rst describe our system model and assumptions in Sec-
tion 3.1, provide a conceptual overview of the algorithm in
Section 3.2, and then present the components ofPlexusin
the remaining sections.
3.1  System Model and Assumptions
We consider a peer-to-peer network ofĤnodes that collabora-
tively train a globalMLmodelĂ. Each participating node has
access to a local dataset which never leaves the participants’
device. Only the model parameters are exchanged between
participating nodes. We assume that each node knows the
specications of the ML model being trained, the learning
hyperparameters, and the settings specic toPlexus. These
specications should be exchanged before training starts.
This work focusses onFLtraining in a cross-device setting
without a central server. Thus, model training withPlexus
proceeds in a decentralized environment and relies on the
## 2

Practical Federated Learning without a ServerEuroMLSys ’25, March 30-April 3 2025, Roerdam, Netherlands
## 1
## 2
## 5
## 4
## 7
## 8
## 6
## 3
Sample k
model
## 1
## 2
## 5
## 4
## 7
## 8
## 6
## 3
## 1
## 2
## 5
## 4
## 7
## 8
## 6
## 3
model
Sample k+1
model
Sample k
Figure 2.Overview of roundġandġ+1inPlexus, with 8 nodes and a sample size of 4 (ĩ=4). Participants are indicated in
blue and the aggregator in green.
cooperation of nodes with varying resource capacities. We
assume that each node has a unique identier (e.g., a public
key) and assume that nodes are connected through a fully
connected overlay network (i.e., all nodes can communicate
with each other). Though the computational capacities of
participating nodes may vary, we assume that each node’s
computational resources are sucient to reliably participate
in learning. We assume that the model being trained ts
into the memory of each node. Also, aggregators inPlexus
should have sucient memory or disk space to store and ag-
gregate the trained models produced by other nodes. While
we remark that nodes might act maliciously during the train-
ing process, we consciously leave out these considerations as
this requires additional mechanisms. However, we acknowl-
edge research in privacy-preserving ML, many of which we
believe could be integrated or adapted intoPlexus[6,9,42].
3.2Plexusin a Nutshell
Similar toFL,Plexus(i)has a subset of nodes (asample)
train the model each round, and(ii)refreshes samples each
round. We refer to nodes belonging to a sample aspartic-
ipants. Among the participants, one node, named theag-
gregator, is responsible for model aggregation during that
round. This aggregator is selected to be the node with the
highest bandwidth capacity as it has to temporarily handle
incoming model transfers from all participants in a round.
In each round, participants are randomly sampled from all
nodes using a consistent hashing scheme. This sampling
mechanism is a key contribution ofPlexusand is further
discussed in Section 3.3.
Figure 2 illustrates two rounds (roundġandġ+1) in
Plexus, with a network containing 8 nodes and a sample
size of 4. We denote the set of nodes in theġ-th sample as
ď
ġ
and the aggregator withinď
ġ
asė
ġ
. At the beginning of
roundġ, the aggregator in the previous sampleď
ġ−1
sends
the aggregated model to all participants inď
ġ
## (step 1). The
participants train the model with their local data and then
send their updated model to aggregatorė
ġ
## (step 2).ė
ġ
## nally
sends the aggregated model to the participants in sample
Algorithm 1Determining a sample and aggregator by node
ğwhereġdenotes the round number andĩis the requested
sample size.
1:procedureSample(ġ,ĩ)
2:Ą←sort([hash(Ġ+ġ)forĠinNodes()])
3:ÿ← [Ġforℎ
## Ġ
inĄ]
## 4:returnÿ[:ĩ]
## 5:
6:procedureAggregator(ġ,ĩ)
## 7:ď
ġ
←Sample(ġ,ĩ)
## 8:
returnĠ∈ď
ġ
such thatĠhas the largest bandwidth among
allď
ġ
nodes according toþ
ğ
ď
ġ+1
, initiating roundġ+1(step 3). This simplied algorithm
description hinges on the ability of nodes to derive samples.
Thus, the main technical challenge lies in deriving consistent
samples in a decentralized fashion.
3.3  Deriving Samples and Aggregators
One of the main novelties ofPlexusis to decentralize the
sampling procedure by having each participant in a sample
compute the next sample independently. In order to achieve
this, each node maintains alocal viewof the network wherein
the membership information of (all) other nodes is recorded.
The gist ofPlexus’s sampling procedure is to rely on a hash
function parameterized by the round number and the node
identiers, stored by all nodes in their local view so that
each node can independently compute the sample of nodes
expected to be active during the training. Algorithm 1 shows
thePlexussampling procedure, which aims to obtain a sam-
ple of
ĩcurrently active nodes in theġ
## Īℎ
round. First, a subset
ofcandidatesis retrieved. Concatenating the node identiers
with round numbers randomizes the order of nodes every
round. The resulting list is sorted in lexicographic order,
which provides the order in which candidates are contacted.
As long as the list of candidates is the same between all
nodes, the order of contact and the resulting samples will
mostly be the same.
## 3

EuroMLSys ’25, March 30-April 3 2025, Roerdam, NetherlandsDhasade et al.
Algorithm 2Training and aggregation by nodeğ.
1:Require: Sample sizeĩ, success fractionĩĜ
2:Θ← []²Listof received models as aggregator
## 3:
4:ifğinsample(1,ĩ)then²Start
training if we are in the rst
sample
## 5:send
toğtrain(1,randomModel())
## 6:
7:upontrain(ġ,Ă)²T
raining
## 8:
## ̄
Ă←train(Ă)
9:ė←Aggregator(ġ,ĩ)²Alg.
## 1
## 10:sendtoėaggregate(ġ,
## ̄
## Ă)
## 11:
12:uponaggregate(ġ,Ă
## Ġ
)fromĠ²A   ggregation
13:Θ.add(Ă
## Ġ
## )
14:ifΘ.sizeg
+ĩ×ĩĜ,then
15:for allĠinď
ġ+1
in parallel do
## 16:Ă
ėĝĝ
←avg(Θ)
17:send toĠtrain(ġ+1,Ă
ėĝĝ
## )
18:Θ← []²Reset
list
We next discuss how participants determine an aggregator.
Internally, this method calls the sampling procedure from
Algorithm 1. The aggregator is a critical node for system
progression and must handle the reception and transmission
of at mostĩtrained models during a round. Since the ag-
gregator has to handle this network load, we preferentially
select the participant with the highest bandwidth capacity
from the derived sample, but acknowledge that other selec-
tion strategies are possible. This biased aggregator selection
optimizes the model transfer times and reduces the time re-
quired per round compared to uniform aggregator selection.
We remark that this biased selection has no impact on the
quality of the trained model. Bandwidth capacities of indi-
vidual nodes can be determined and synchronized a-priori
to training. We found that this decision was essential to the
success ofPlexusas learning progress would slow down
signicantly if an aggregator with low bandwidth capacity
is chosen, especially if the model size increases.
3.4  Training and Aggregating Models
Each node inPlexusimplements two procedures: one for
aggregation and one for training. We provide the pseudocode
of these procedures in Algorithm 2. The design ofPlexus
is based on apush-based architecture, in which nodes in
sampleď
ġ
trigger the activation of nodes in sampleď
ġ+1
## .
This way, nodes do not have to continuously be aware of the
current training round being worked on; they only have to
start working when receiving a trained or aggregated model.
Training.We rst describe the training procedure by
nodeğ. A node starts the training task when it receives a
trainmessage containing an aggregated modelĂand a
round number
ġ.ğtrains the received modelĂby calling
thetrainprocedure, resulting in trained model
## ̄
## Ă. Then,
ğdetermines aggregatorėin current sampleġby calling
theaggregateprocedure. This aggregator is derived using
our peer sampler (see Section 3.3 and Algorithm 1). Finally,ğ
sends anAggregatemessage, containing the resulting model
## ̄
Ă, to aggregatorė.
Aggregation.We provide the pseudocode related to ag-
gregation in Algorithm 2. An aggregatorėstarts the aggrega-
tion task when it is activated through anaggregatemessage
from nodeĠ, containing trained modelĂ
## Ġ
.ėkeeps track of
the received models in listΘand addsĂ
## Ġ
toΘ. Upon receiv-
ing at least+ĩ×ĩĜ,models for roundġ,ėaggregates these
models, resulting inĂ
ėĝĝ
. We refer to the required fraction of
models needed as thesuccess fractionĩĜ. This oversampling
is common in realisticFLsystems [1].ėthen determines the
participants in sampleġ+1and sends these nodes atrain
message containing the next round numberġ+1and the
aggregated modelĂ
ėĝĝ
. This completes roundġ.
3.5Plexusand FL
PlexusbringsFLto fully decentralized networks. We now
discuss how the elements ofPlexustranslate to those of
FL. Firstly, the local model training at the nodes inPlexus
is equivalent to that inFL. The aggregator inPlexustem-
porarily performs the role of the FL server, aggregating the
model updates in the current round. The central server in
FLalso orchestrates training by choosing the participants in
each round. This role is handled by the decentralized sam-
pling mechanism inPlexusdescribed in Section 3.3. As a
consequence of the similarity ofPlexusto
FL, the conver-
gence proofs forFLalso oer theoretical grounding for the
convergence of our approach [34].
## 4  Experimental Evaluation
We now present the experimental evaluation ofPlexus. We
provide all relevant details on our experiment setup in Sec-
tion 4.1. Our evaluation answers the following two questions:
•What is the performance ofPlexusin terms of wall-
clock convergence time, communication volume and
training resource usage compared toFLandDLbase-
lines (Section 4.2)?
•What is the eect of the sample sizeĩinPlexuson
wall-clock convergence time, communication volume
and training resource usage (Section 4.3)?
## 4.1  Experiment Setup
We implementPlexusin the Python 3 programming lan-
guage.
## 1
Plexusleverages theIPv8networking library which
provides support for authenticated messaging and build-
ing decentralized overlay networks [
59]. Our implementa-
tion adopts an event-driven programming model with the
asynciolibrary. We use thePyTorchlibrary [47] to train ML
models, and the dataset API fromDecentralizePy[17]. As
## 1
Source code:hps://github.com/sacs-epfl/plexus.
## 4

Practical Federated Learning without a ServerEuroMLSys ’25, March 30-April 3 2025, Roerdam, Netherlands
PlexusFLGLD−PSGD (OP)D−PSGD (k−reg)
## 30
## 40
## 50
## 60
## 70
## 01020304050
Time into Experiment [h]
## Test Acc. [%]
## 70
## 75
## 80
## 85
## 90
## 01020304050
Time into Experiment [h]
## Test Acc. [%]
## 20
## 40
## 60
## 80
## 050100150200
Time into Experiment [h]
## Test Acc. [%]
## 30
## 40
## 50
## 60
## 70
## 110100100010000
Communication Volume [GiB]
## Test Acc. [%]
## 70
## 75
## 80
## 85
## 90
## 1101001000
Communication Volume [GiB]
## Test Acc. [%]
## 20
## 40
## 60
## 80
## 100100010000
Communication Volume [GiB]
## Test Acc. [%]
## 30
## 40
## 50
## 60
## 70
## 10100100010000
## Training Resource Usage [h]
## Test Acc. [%]
(a)CIFAR-10 (IID)
## 70
## 75
## 80
## 85
## 90
## 10100100010000
## Training Resource Usage [h]
## Test Acc. [%]
(b)CelebA (non-IID)
## 20
## 40
## 60
## 80
## 10100100010000
## Training Resource Usage [h]
## Test Acc. [%]
(c)FEMNIST(non-IID)
Figure 3.The convergence ofPlexusand baselines against time (top), communication volume (middle), and training resource
usage (bottom).
a node might be involved in multiple incoming and outgoing
model transfers simultaneously, we equip each node with a
bandwidth scheduler that we implemented. This scheduler
coordinates all model transfers a particular node is involved
in and enables us to realistically emulate the duration of
model transfers.
Hardware.We run all experiments on machines in our na-
tional compute cluster. Each machine is equipped with dual
24-core AMD EPYC-2 CPU, 128 GB of memory, an NVIDIA
RTX A4000 GPU, and is running CentOS 8. Similar to related
work in the domain, we simulate the passing of time in our
experiments [1,32,33]. We achieve this by customizing the
default event loop provided by theasynciolibrary and pro-
cessing events without delay. The real-world time required
to reproduce our experiment depends on the baseline being
evaluated. Running ourPlexusand ourFLbaseline requires
between 6 hours (for CelebA) and 24 hours (for FEMNIST)
of compute time. TheDLbaselines are more computation-
ally intense since they have every node training each round
and require between 24 hours (for CelebA) and 5 days (for
FEMNIST) of compute time. Our simulator is implemented
as a single process and its eciency can be further improved
by paralellizing the training.
Traces.We have designedPlexusto operate in highly
heterogeneous environments, such as mobile networks. To
verify thatPlexusalso functions in such environments, we
adopt various real-world traces to simulate pairwise network
Table 1.Summary of datasets and learning parameters used
to evaluatePlexusandDLbaselines. “Mom.” denotes the
momentum parameter.
DatasetNodesLearning ParametersModel
CIFAR-10 [31]1000ā=0.002,mom.=0.9CNN [22]
CelebA [12]500ā=0.001CNN
FEMNIST[12]355ā=0.004CNN
latency, bandwidth capacities, compute speed, and availabil-
ity. To model a WAN environment, we apply latency to out-
going network trac at the application layer to realistically
model delays in sendingPlexuscontrol messages. To this
end, we collect ping times from WonderNetwork, providing
estimations on the RTT between their servers located in 227
geo-separated cities [62]. When starting an experiment, we
assign peers to each city in a round-robin fashion and delay
outgoing network trac accordingly.
We also adopt traces from theFedScalebenchmark to
simulate the hardware performance of nodes, specically
network and compute capacities [32]. These traces contain
the hardware prole of131 000mobile devices and are origi-
nally sourced from the AI benchmark [26] and theMobiPerf
measurements [23]. We assume that nodes are aware of the
bandwidth capabilities of other nodes, and within a sample, a
node sends its trained model to the aggregator with the high-
est bandwidth capability in the next sample. In summary,
## 5

EuroMLSys ’25, March 30-April 3 2025, Roerdam, NetherlandsDhasade et al.
our experiments go beyond existing work onDLby inte-
grating multiple traces that together account for the system
heterogeneity in WAN environments.
Datasets.We evaluatePlexuson dierent models and
with three distinct datasets, whose characteristics are dis-
played in Table 1. The CIFAR-10 dataset [31] is IID, parti-
tioned by uniformly randomly assigning data samples to
nodes. The CelebA and FEMNIST datasets are taken from
the LEAF benchmark [12], which was specically designed
to evaluate the performance of learning tasks in non-IID
settings. The sample-to-node assignment for CelebA and
FEMNIST is given by the LEAF benchmark. Our evaluation,
thus, covers a variety of learning tasks and data partitions.
Performance Metrics and Hyperparameters.We mea-
sure the top-1 test accuracy of the model on a global test
set unseen during training, for the purpose of evaluation.
In line with other work, we x the batch size to 20 for all
experiments and each device always performs ve local steps
when training its model [1,32] before communicating. All
models are trained using the SGD optimizer. For CIFAR-10
we additionally use a momentum factor of0.9. All our learn-
ing parameters were adopted from previous works [4,12]
or were considered after trials on several values. They yield
acceptable target accuracy for all evaluated datasets. We run
each experiment three times with dierent seeds and report
averaged values.
Baselines.We comparePlexusagainst aFLsetup in
which we assume the availability of a server with unlimited
bandwidth constraints. We also use Gossip learning (GL) [21]
andD-PSGD[35] asDLbaselines. In each round ofGL, a
node rst waits for some time and then sends its model to an-
other random node in the network. The selection of nodes is
facilitated by a peer-sampling service which presents a view
of random nodes in the network every round. Upon receiv-
ing a model from another node, the recipient node merges it
with its own local model, weighted by the model age, and
trains the local model.GLnaturally tolerates churn and is ro-
bust to failing nodes. However, pairwise model aggregation
still leaves residual variance and deteriorates model conver-
gence compared to when using global aggregation. In our
experiments, we x the round timeout to60seconds forGL
to give each node sucient time to train and transfer the
model each local round.
## D-PSGD
[35] is a synchronous algorithm that only pro-
ceeds when all nodes have received all models from their
neighboring nodes. We evaluateD-PSGDunder two topolo-
gies: a 10-regular topology (i.e., each node has ten neighbors)
and a one-peer exponential graph topology, the latter being
a state-of-the-art topology inDL[64]. Thus, we evaluate
D-PSGD with sparse and dense graph connectivity.
ForPlexus, we report the accuracy of the global model
after aggregation every ten rounds. ForD-PSGD, we deter-
mine the mean and standard deviation of the accuracy ob-
tained by evaluating models of individual nodes on the test
dataset every two hours. We also report communication vol-
ume (transmitted bytes) and training resource usage (i.e., the
time a device spends on model training). ForPlexus, we set
ĩ=13and adjust the aggregator so it proceeds when it has
received 80% of all models (ĩĜ=0.8). We run experiments
with CIFAR-10 and CelebA for 50 hours and FEMNIST for
200 hours which gives model training withPlexusand other
baselines sucient time to converge.
4.2PlexusCompared to Baselines
We quantify and compare the performance ofPlexuswith
baseline systems. We comparePlexusagainst aFLsetup in
which we assume the availability ofa server with unlimited
bandwidth constraints. We also use Gossip learning (GL) [21]
andD-PSGD[35] asDLbaselines. We evaluateD-PSGDun-
der two topologies: a 10-regular topology (i.e., each node
has ten neighbors) and a one-peer exponential graph topol-
ogy, the latter being a state-of-the-art topology inDL[64].
Thus, we evaluateD-PSGDwith both sparse and dense graph
connectivity.
Results.We show the performance ofPlexusand base-
lines in Figure 3. The top row of Figure 3 shows the test accu-
racy as the experiment progresses.Plexusoutperforms both
DLbaselines by converging quicker and achieving higher
test accuracy, consistently across all datasets. In general, we
nd that inGLmore training occurs within a given time unit
compared toDL, sinceGLrounds are asynchronous and in-
dividual nodes have less idle time compared toD-PSGD. On
the simpler binary classication task for the CelebA dataset,
the performance improvements ofPlexusare modest. How-
ever, on more dicult learning tasks like the 62-class image
classication in FEMNIST with a larger model size,Plexus
achieves more than 20% better accuracy when compared to
the best performingDLbaseline,GL. We also observe that
Plexusgenerally shows comparable time-to-accuracy asFL
but with the larger model size of FEMNIST we observe small
dierences since the server inFLhas unlimited bandwidth.
The middle row in Figure 3 shows the communication vol-
ume (horizontal axis in log scale) required to achieve the test
accuracy for the evaluated systems.Plexusattains high test
accuracies with orders of magnitude less transmitted bytes.
These eciency gains are particularly pronounced for the
FEMNIST dataset. We note thatD-PSGDwith a 10-regular
topology incurs the most network trac while performing
on par or worse than the one-peer exponential graph topol-
ogy. The bottom row in Figure 3 shows the training resource
usage (horizontal axis in log scale) consumed to achieve
the test accuracy.Plexusattains high test accuracies with
orders of magnitude less resource usage. Finally, we note
thatPlexusincurs comparable communication volume and
resource usage as the centralized baseline of FL.
For each dataset, we determine the best-performing base-
line in terms of the highestindividual model accuracy achieved
across all nodes. We then determine time-to-accuracy (TTA),
## 6

Practical Federated Learning without a ServerEuroMLSys ’25, March 30-April 3 2025, Roerdam, Netherlands
## Sample Size
s=10s=20s=40
## 20
## 40
## 60
## 80
## 0204060
Time into Experiment [h]
## Test Accuracy
(a)Model convergence
## 20
## 40
## 60
## 80
## 301003001000
Communication Volume [GiB]
## Test Accuracy
(b)CommunicationVolume
## 20
## 40
## 60
## 80
## 101001000
## Training Resource Usage [h]
## Test Accuracy
(c)Training Resource Usage
Figure 4.The performance ofPlexusfor dierent sample sizesĩ, using the FEMNIST dataset.
## 40
## 20
## 10
## 50100150
## Round Duration [s]
## Sample Size
Figure 5.The distribution of round durations, for the FEM-
NIST dataset and dierent sample sizes.
communication-to-accuracy (CTA), and resources-to-accuracy
(RTA), which are metrics that represent the ecacy, e-
ciency, and scalability ofDLsystems. For the evaluated
datasets and compared to the target accuracy,Plexussaves
1.4×- 1.6×in TTA, 15.8×- 292×in CTA and 30.5×- 77.9×in
RTA compared toGLandD-PSGD. In conclusion, our com-
prehensive evaluation demonstrates the superior eciency
and eectiveness ofPlexus.
4.3  Varying the Sample Sizeĩ
We next explore the eect of the sample sizeĩon model
convergence, communication volume, training resource us-
age, and round duration by runningPlexuswithĩ=10,20,
and40. This experiment uses the FEMNIST dataset which
has the largest model size in our setup. Figure 4a shows the
test accuracy for the three dierent values ofĩas the ex-
periment progresses. We observe that increasingĩactually
slows down training, likely because more models have to be
transferred to and from the aggregator. Naturally, increasing
ĩalso has a negative impact on communication cost and
resource usage, which are visualized in Figure 4b and Fig-
ure 4c, respectively. In particular, reaching 80% test accuracy
forĩ=40incurs 4.0×additional communication volume and
4.0×more training resource usage, when compared toĩ=10.
We note, however, that increasingĩmight be favorable in
other scenarios,e.g., when data is highly non-IID.
Increasingĩalso prolongs the duration of individual rounds
as the aggregator has to receive and redistribute more mod-
els. We show in Figure 5 the distribution of round durations
in seconds for dierent values ofĩusing a box and violin plot.
When increasingĩfrom 10 to 40, the average round duration
increases from45.1seconds. to54.9seconds. Moreover, we
observe that some rounds take disproportionally long which
can happen when all nodes in a selected sample have low
bandwidth or slow compute speeds. For example, a round
for
ĩ=20took up up to178seconds. However, this is rela-
tively rare: forĩ=20, only 18 rounds out of6941took over
100secondsto complete. We observe also a positive eect on
round duration when increasingĩ: with higher values ofĩ,
there is a higher probability that nodes with high bandwidth
capacities are included in the sample compared to lower
values of
ĩ, which lowers the overall model transfer times
during a round. We can see this eect in Figure 5 as there is
a higher variance in round durations for lower values ofĩ.
Empirically, we obtain a good trade-o between sample size
and convergence when setting the sample size around 10.
## 5  Related Work
Decentralized Learning (DL).D-PSGD[35] showed theo-
retically and empirically that under strong bandwidth lim-
itations on an aggregation server in a data center, decen-
tralized algorithms can converge faster. Assumptions on
the behavior of those algorithms make them most suited
to data centers. The synchronization required inD-PSGD
is costly when training on edge devices. As a solution, re-
search inDLhas been focussed either on having a better
topology [4,16,54,56,60,64], or designing asynchronous
algorithms [7,37,43,46]. MoshpitSGD [50] uses a DHT to
randomly combine nodes in multiple disjoint cohorts every
round for fast-averaging convergence. Notably, Teleporta-
tion is aDLalgorithm where, similar toPlexus, a small
subset of nodes train every round and then exchange models
with other sampled nodes over a smaller topology [57]. How-
ever, the paper does not specify exactly how these nodes
are sampled andPlexussolves this issue through consistent
hashing. All the above algorithms, however, overlook system
heterogeneity whereas we evaluatedPlexusin the presence
of network and compute heterogeneity.
Federated Learning (FL).FLis arguably the most pop-
ular algorithm for privacy-preserving distributed learning
## 7

EuroMLSys ’25, March 30-April 3 2025,Roerdam, NetherlandsDhasade et al.
and uses a parameter server to coordinate the learning pro-
cess [41]. Similar toPlexus, FL lowers resource and com-
munication costs at the edge by having a small subset of
nodes train the model every round. To make FL suitable at
scale in deployment scenarios, recent works have placed sig-
nicant emphasis on system challenges [1,8,25,33,45,66].
FL still requires a highly available central server that can
support many clients concurrently, possibly resulting in high
infrastructure costs.Plexus, on the contrary, is a fully de-
centralized system with an aggregation scheme inspired by
FL, while avoiding central coordination.
Blockchain-AssistedDL.We identied various works
that implement and evaluate blockchain-based decentralized
learning systems [3,30,39,40,48,61] and discuss the chal-
lenges therein [58,67]. Consensus-based replicated ledgers
used in these systems provide strong consensus primitives at
the cost of a signicant and unnecessary overhead. Machine
learning optimizations based on SGD thrive in the presence
of stochasticity, obviating the need for strong consensus in
the form of a global model [36, 46].
Decentralized Peer Sampling.Brahms [10], Basalt [2],
PeerSampling [27] and PeerSwap [19] provide each node
with a dierent, uniformly random sample without network-
wide synchronization. In contrast, the peer sampling ofPlexus
instead ensures that nodes selectequivalentsamples in a par-
ticular round of training.
6  Broader Impact and Open Challenges
The broader impact of our work is multifaceted, addressing
both a technical and socio-economic dimension. By circum-
venting the need for a central server,Plexuscan lower the
barrier and costs to adoptFL-like training in decentralized
settings. A complementary benet ofPlexusis that it en-
hances data privacy and security by decentralizing the model
aggregation process, reducing the risk of data breaches asso-
ciated with centralized systems. This is because, depending
on the total network size and sample size, it becomes more
dicult for a single node to systematically access model up-
dates from particular nodes, raising the barrier for privacy
attacks such as the gradient inversion attack [24].
Open Challenges.We discuss several open challenges in
the current design ofPlexus. Like any decentralized system,
Plexusmust balance eciency with network constraints.
As model sizes increase, aggregators may experience heavier
loads, potentially prolonging round durations. This also ap-
plies toFLwith a server, although it is typical that the server
coordinating theFLprocess has more compute capacity than
the devices participating in the training process. To account
for this bottleneck inPlexus, we select the aggregator as the
node with the highest bandwidth capabilities within a sam-
ple. However, in rare cases, all nodes in a sample may have
low bandwidth capabilities due to an unfortunate selection
(also see Section 4.3). In this scenario, we could select an ag-
gregator from outside the current sample. Another approach
to reduce communication burdens on aggregators is to use
multiple aggregators in the same round. Both enhancements,
however, require us to deviate from the standardFLalgo-
rithm and necessitate additional coordination mechanisms.
Secondly, we currently assume all nodes remain online
during training which is typically not the case in real-world
settings. Our idea is to extendPlexuswith support for churn
by including the current online or oine status in the local
views, and synchronize these views between samples. How-
ever, dealing with churn requires additional coordination as
nodes can also go oine during training or aggregation.
Thirdly,Plexusoverutilizes the computational resources.
In our experiments, the round completes when 80% of the
models are received, thus at least 20% of trained models will
never be aggregated. As a result, some participant updates
may not be aggregated in every round, similar to commonFL
techniques that mitigate stragglers [8]. VariousFLsystems
alleviate this issue by integrating stale model updates [1,15,
63]. We leave integrating this inPlexusfor future work.
Finally, securingPlexusagainst Byzantine nodes remains
an open challenge. A promising direction is to incorporate
accountability mechanisms where nodes verify the integrity
of sample selection, ensuring honest participation.
Overall, these aspects highlight common challenges in
FLandDL. Addressing them will improve the robustness
and reliability ofPlexus, paving the way for deployment
of decentralized learning systems in open and large-scale
settings.
## 7  Conclusions
This paper introducedPlexus, a practical, ecient and decen-
tralizedFLsystem. The two key components of our system
are(i)a decentralized peer sampler to select small subsets
of nodes each round, and(ii)a global aggregation opera-
tion within these selected subsets. Extensive evaluations
with realistic traces of compute speed, network capacity,
and availability in decentralized networks up to1000nodes
demonstrate the superiority ofPlexusover baselineDL
algorithms, reducing time-to-accuracy by 1.4-1.6×, commu-
nication volume by 15.8-292×, and training resource usage
by 30.5-77.9×compared toDLbaselines. Moreover,Plexus
also achieves accuracy and resource usage comparable to a
centralized FL baseline.
## Acknowledgments
This work has been funded by the Swiss National Science
Foundation, under the project “FRIDAY: Frugal, Privacy-
Aware and Practical Decentralized Learning”, SNSF proposal
No. 10.001.796. This work has also been funded by the Dutch
national NWO/TKI science grant BLOCK.2019.004.
## 8

Practical Federated Learning without a ServerEuroMLSys ’25, March 30-April 3 2025, Roerdam, Netherlands
## References
[1]Ahmed M Abdelmoniem, Atal Narayan Sahu, Marco Canini, and
Suhaib A Fahmy. Re: Resource-ecient federated learning. InPro-
ceedings of the Eighteenth European Conference on Computer Systems,
pages 215–232, 2023.
[2]Alex Auvolat, Yérom-David Bromberg, Davide Frey, and François
Taïani.  Basalt: A rock-solid foundation for epidemic consensus
algorithms in very large, very open networks.arXiv preprint
arXiv:2102.04063, 2021.
[3]Xianglin Bao, Cheng Su, Yan Xiong, Wenchao Huang, and Yifei Hu.
Flchain: A blockchain for auditable federated learning with trust and
incentive. In2019 5th International Conference on Big Data Computing
and Communications (BIGCOM), pages 151–159. IEEE, 2019.
[4]Aurélien Bellet, Anne-Marie Kermarrec, and Erick Lavoie. D-cliques:
Compensating for data heterogeneity with topology in decentralized
federated learning. In2022 41st International Symposium on Reliable
Distributed Systems (SRDS), pages 1–11. IEEE, 2022.
[5]Enrique Tomás Martínez Beltrán, Mario Quiles Pérez, Pedro
## Miguel Sánchez Sánchez, Sergio López Bernal, Gérôme Bovet,
Manuel Gil Pérez, Gregorio Martínez Pérez, and Alberto Huertas Cel-
drán. Decentralized federated learning: Fundamentals, state of the art,
frameworks, trends, and challenges.IEEE Communications Surveys &
## Tutorials, 2023.
[6]Sayan Biswas, Mathieu Even, Anne-Marie Kermarrec, Laurent Mas-
soulié, Rafael Pires, Rishi Sharma, and Martijn de Vos.  Noiseless
privacy-preserving decentralized learning.Proceedings on Privacy
## Enhancing Technologies, 2025.
## [7]
Sayan Biswas, Anne-Marie Kermarrec, Alexis Marouani, Rafael Pires,
Rishi Sharma, and Martijn de Vos. Boosting Asynchronous Decentral-
ized Learning with Model Fragmentation. InProceedings of the ACM
on Web Conference 2025, 2025.
[8]Keith Bonawitz, Hubert Eichner, Wolfgang Grieskamp, Dzmitry Huba,
## Alex Ingerman, Vladimir Ivanov, Chloe Kiddon, Jakub Konečn
## `
y, Ste-
fano Mazzocchi, Brendan McMahan, et al. Towards federated learning
at scale: System design.Proceedings of machine learning and systems,
## 1:374–388, 2019.
[9]Keith Bonawitz, Vladimir Ivanov, Ben Kreuter, Antonio Marcedone,
H Brendan McMahan, Sarvar Patel, Daniel Ramage, Aaron Segal, and
Karn Seth. Practical secure aggregation for privacy-preserving ma-
chine learning. Inproceedings of the 2017 ACM SIGSAC Conference on
Computer and Communications Security, pages 1175–1191, 2017.
[10]Edward Bortnikov, Maxim Gurevich, Idit Keidar, Gabriel Kliot, and
Alexander Shraer. Brahms: Byzantine resilient random membership
sampling.Computer Networks, 53(13):2340–2359, 2009.
## [11]
## Theodora S Brisimi, Ruidi Chen, Theofanie Mela, Alex Olshevsky,
Ioannis Ch Paschalidis, and Wei Shi. Federated learning of predictive
models from federated electronic health records.International journal
of medical informatics, 112:59–67, 2018.
[12]Sebastian Caldas, Sai Meher Karthik Duddu, Peter Wu, Tian Li, Jakub
## Konečn
## `
y, H Brendan McMahan, Virginia Smith, and Ameet Talwalkar.
Leaf: A benchmark for federated settings. In2nd Intl. Workshop on
Federated Learning for Data Privacy and Condentiality (FL-NeurIPS),
## 2019.
## [13]
Mingqing Chen, Rajiv Mathews, Tom Ouyang, and Françoise Beau-
fays. Federated learning of out-of-vocabulary words.arXiv preprint
arXiv:1903.10635, 2019.
## [14]
## Mingqing Chen, Ananda Theertha Suresh, Rajiv Mathews, Ade-
line Wong, Cyril Allauzen, Françoise Beaufays, and Michael Riley.
Federated learning of n-gram language models.arXiv preprint
arXiv:1910.03432, 2019.
[15]Georgios Damaskinos, Rachid Guerraoui, Anne-Marie Kermarrec, Vlad
Nitu, Rhicheek Patra, and Francois Taiani. Fleet: Online federated
learning via staleness awareness and performance prediction.ACM
Transactions on Intelligent Systems and Technology (TIST), 13(5):1–30,
## 2022.
[16]Martijn De Vos, Sadegh Farhadkhani, Rachid Guerraoui, Anne-Marie
Kermarrec, Rafael Pires, and Rishi Sharma. Epidemic learning: Boost-
ing decentralized learning with randomized communication.Advances
in Neural Information Processing Systems, 36, 2023.
[17]Akash Dhasade, Anne-Marie Kermarrec, Rafael Pires, Rishi Sharma,
and Milos Vujasinovic. Decentralized learning made easy with de-
centralizepy. InProceedings of the 3rd Workshop on Machine Learning
and Systems, EuroMLSys ’23, page 34–41, New York, NY, USA, 2023.
Association for Computing Machinery.
## [18]
Anousheh Gholami, Nariman Torkzaban, and John S. Baras. Trusted
decentralized federated learning. In2022 IEEE 19th Annual Consumer
Communications and Networking Conference (CCNC), pages 1–6, 2022.
[19]Rachid Guerraoui, Anne-Marie Kermarrec, Anastasiia Kucherenko,
Rafael Pinot, and Martijn de Vos. Peerswap: A peer-sampler with
randomness guarantees. In2024 43rd International Symposium on
Reliable Distributed Systems (SRDS), pages 271–281. IEEE, 2024.
[20]Andrew Hard, Kanishka Rao, Rajiv Mathews, Swaroop Ramaswamy,
## Françoise Beaufays, Sean Augenstein, Hubert Eichner, Chloé Kiddon,
and Daniel Ramage. Federated learning for mobile keyboard prediction.
arXiv preprint arXiv:1811.03604, 2018.
[21]István Hegedűs, Gábor Danner, and Márk Jelasity. Gossip learning
as a decentralized alternative to federated learning. InDistributed
Applications and Interoperable Systems: 19th IFIP WG 6.1 International
Conference, DAIS 2019, Held as Part of the 14th International Federated
Conference on Distributed Computing Techniques, DisCoTec 2019, Kon-
gens Lyngby, Denmark, June 17–21, 2019, Proceedings 19, pages 74–90.
## Springer, 2019.
## [22]
Kevin Hsieh, Amar Phanishayee, Onur Mutlu, and Phillip Gibbons.
The non-iid data quagmire of decentralized machine learning.  In
International Conference on Machine Learning, pages 4387–4398. PMLR,
## 2020.
[23]Junxian Huang, Cheng Chen, Yutong Pei, Zhaoguang Wang, Zhiyun
Qian, Feng Qian, Birjodh Tiwana, Qiang Xu, Z Mao, Ming Zhang, et al.
Mobiperf: Mobile network measurement system.Technical Report.
University of Michigan and Microsoft Research, 2011.
[24]Yangsibo Huang, Samyak Gupta, Zhao Song, Kai Li, and Sanjeev Arora.
Evaluating gradient inversion attacks and defenses in federated learn-
ing.Advances in Neural Information Processing Systems, 34:7232–7241,
## 2021.
## [25]
## Dzmitry Huba, John Nguyen, Kshitiz Malik, Ruiyu Zhu, Mike Rabbat,
Ashkan Yousefpour, Carole-Jean Wu, Hongyuan Zhan, Pavel Ustinov,
Harish Srinivas, Kaikai Wang, Anthony Shoumikhin, Jesik Min, and
Mani Malek. Papaya: Practical, private, and scalable federated learning.
In D. Marculescu, Y. Chi, and C. Wu, editors,Proceedings of Machine
Learning and Systems, volume 4, pages 814–832, 2022.
[26]Andrey Ignatov, Radu Timofte, Andrei Kulik, Seungsoo Yang, Ke Wang,
Felix Baum, Max Wu, Lirong Xu, and Luc Van Gool. Ai benchmark:
All about deep learning on smartphones in 2019. In2019 IEEE/CVF
International Conference on Computer Vision Workshop (ICCVW), pages
## 3617–3635. IEEE, 2019.
[27]Márk Jelasity, Spyros Voulgaris, Rachid Guerraoui, Anne-Marie Ker-
marrec, and Maarten Van Steen. Gossip-based peer sampling.ACM
Transactions on Computer Systems (TOCS), 25(3):8–es, 2007.
[28]Jiawen Kang, Zehui Xiong, Dusit Niyato, Yuze Zou, Yang Zhang, and
Mohsen Guizani. Reliable federated learning for mobile networks.
IEEE Wireless Communications, 27(2):72–80, 2020.
[29]Lingjing Kong, Tao Lin, Anastasia Koloskova, Martin Jaggi, and Se-
bastian Stich. Consensus control for decentralized deep learning. In
Marina Meila and Tong Zhang, editors,Proceedings of the 38th Interna-
tional Conference on Machine Learning, volume 139 ofProceedings of
Machine Learning Research, pages 5686–5696. PMLR, 18–24 Jul 2021.
## [30]
## Caner Korkmaz, Halil Eralp Kocas, Ahmet Uysal, Ahmed Masry, Oznur
Ozkasap, and Baris Akgun. Chain : Decentralized federated machine
## 9

EuroMLSys ’25, March 30-April 3 2025, Roerdam, NetherlandsDhasade et al.
learning via blockchain. In2020 Second International Conference on
Blockchain Computing and Applications (BCCA), pages 140–146, 2020.
[31]Alex Krizhevsky. Learning Multiple Layers of Features from Tiny
Images.hps://www.cs.toronto.edu/~kriz/learning-features-2009-TR.
pdf, 2009.
[32]Fan Lai, Yinwei Dai, Sanjay Singapuram, Jiachen Liu, Xiangfeng Zhu,
Harsha Madhyastha, and Mosharaf Chowdhury.  Fedscale: Bench-
marking model and system performance of federated learning at scale.
InInternational Conference on Machine Learning, pages 11814–11827.
## PMLR, 2022.
## [33]
Fan Lai, Xiangfeng Zhu, Harsha V. Madhyastha, and Mosharaf Chowd-
hury. Oort: Ecient federated learning via guided participant selection.
In15th USENIX Symposium on Operating Systems Design and Imple-
mentation (OSDI 21), pages 19–35. USENIX Association, July 2021.
## [34]
Xiang Li, Kaixuan Huang, Wenhao Yang, Shusen Wang, and Zhihua
Zhang. On the convergence of fedavg on non-iid data. InInternational
Conference on Learning Representations, 2019.
[35]Xiangru Lian, Ce Zhang, Huan Zhang, Cho-Jui Hsieh, Wei Zhang,
and Ji Liu. Can decentralized algorithms outperform centralized al-
gorithms? a case study for decentralized parallel stochastic gradient
descent.Advances in Neural Information Processing Systems, 30, 2017.
[36]Xiangru Lian, Ce Zhang, Huan Zhang, Cho-Jui Hsieh, Wei Zhang, and
## Ji Liu. Can Decentralized Algorithms Outperform Centralized Algo-
rithms? A Case Study for Decentralized Parallel Stochastic Gradient
Descent. InNIPS, 2017.
[37]Xiangru Lian, Wei Zhang, Ce Zhang, and Ji Liu. Asynchronous de-
centralized parallel stochastic gradient descent. In Jennifer Dy and
Andreas Krause, editors,Proceedings of the 35th International Con-
ference on Machine Learning, volume 80 ofProceedings of Machine
Learning Research, pages 3043–3052. PMLR, 10–15 Jul 2018.
## [38]
Songtao Lu, Yawen Zhang, and Yunlong Wang. Decentralized federated
learning for electronic health records. In2020 54th Annual Conference
on Information Sciences and Systems (CISS), pages 1–5. IEEE, 2020.
[39]Yunlong Lu, Xiaohong Huang, Yueyue Dai, Sabita Maharjan, and Yan
Zhang. Blockchain and federated learning for privacy-preserved data
sharing in industrial iot.IEEE Transactions on Industrial Informatics,
## 16(6):4177–4186, 2019.
[40]Umer Majeed and Choong Seon Hong. Flchain: Federated learning via
mec-enabled blockchain network. In2019 20th Asia-Pacic Network
Operations and Management Symposium (APNOMS), pages 1–4. IEEE,
## 2019.
## [41]
Brendan McMahan, Eider Moore, Daniel Ramage, Seth Hampson, and
Blaise Aguera y Arcas. Communication-Ecient Learning of Deep
Networks from Decentralized Data. In Aarti Singh and Jerry Zhu,
editors,Proceedings of the 20th International Conference on Articial
Intelligence and Statistics, volume 54 ofProceedings of Machine Learning
Research, pages 1273–1282. PMLR, 20–22 Apr 2017.
[42]Viraaji Mothukuri, Reza M Parizi, Seyedamin Pouriyeh, Yan Huang,
Ali Dehghantanha, and Gautam Srivastava. A survey on security and
privacy of federated learning.Future Generation Computer Systems,
## 115:619–640, 2021.
[43]Adel Nabli, Eugene Belilovsky, and Edouard Oyallon. A2CiD2: Acceler-
ating Asynchronous Communication in Decentralized Deep Learning.
Advances in Neural Information Processing Systems, 36, 2023.
[44]Dinh C Nguyen, Quoc-Viet Pham, Pubudu N Pathirana, Ming Ding,
Aruna Seneviratne, Zihuai Lin, Octavia Dobre, and Won-Joo Hwang.
Federated learning for smart healthcare: A survey.ACM Computing
Surveys (Csur), 55(3):1–37, 2022.
[45]John Nguyen, Kshitiz Malik, Hongyuan Zhan, Ashkan Yousefpour,
Mike Rabbat, Mani Malek, and Dzmitry Huba. Federated learning
with buered asynchronous aggregation.  In Gustau Camps-Valls,
Francisco J. R. Ruiz, and Isabel Valera, editors,Proceedings of The 25th
International Conference on Articial Intelligence and Statistics, volume
151 ofProceedings of Machine Learning Research, pages 3581–3607.
PMLR, 28–30 Mar 2022.
[46]Róbert Ormándi, István Hegedűs, and Márk Jelasity. Gossip learn-
ing with linear models on fully distributed data.Concurrency and
Computation: Practice and Experience, 25(4):556–571, 2013.
[47]Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Brad-
bury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein,
Luca Antiga, et al. Pytorch: An imperative style, high-performance
deep learning library.Advances in neural information processing sys-
tems, 32, 2019.
[48]Shiva Raj Pokhrel and Jinho Choi. Federated learning with blockchain
for autonomous vehicles: Analysis and design challenges.IEEE Trans-
actions on Communications, 68(8):4734–4746, 2020.
[49]Nicola Rieke, Jonny Hancox, Wenqi Li, Fausto Milletari, Holger R
## Roth, Shadi Albarqouni, Spyridon Bakas, Mathieu N Galtier, Bennett A
Landman, Klaus Maier-Hein, et al. The future of digital health with
federated learning.NPJ digital medicine, 3(1):1–7, 2020.
[50]Max Ryabinin, Eduard Gorbunov, Vsevolod Plokhotnyuk, and Gennady
Pekhimenko. Moshpit sgd: Communication-ecient decentralized
training on heterogeneous unreliable devices.Advances in Neural
## Information Processing Systems, 34, 2021.
## [51]
Mikael Sabuhi, Petr Musilek, and Cor-Paul Bezemer. Micro-: A fault-
tolerant scalable microservice-based platform for federated learning.
## Future Internet, 16(3):70, 2024.
[52]William Schneble and Geethapriya Thamilarasu. Attack detection
using federated learning in medical cyber-physical systems. InProc.
28th Int. Conf. Comput. Commun. Netw.(ICCCN), volume 29, pages 1–8,
## 2019.
[53]Khe Chai Sim, Françoise Beaufays, Arnaud Benard, Dhruv Guliani,
## Andreas Kabel, Nikhil Khare, Tamar Lucassen, Petr Zadrazil, Harry
Zhang, and Leif Johnson. Personalization of end-to-end speech recog-
nition on mobile devices for named entities. In2019 IEEE Automatic
Speech Recognition and Understanding Workshop (ASRU), pages 23–30.
## IEEE, 2019.
[54]Zhuoqing Song, Weijian Li, Kexin Jin, Lei Shi, Ming Yan, Wotao Yin,
and Kun Yuan. Communication-ecient topologies for decentralized
learning with $o(1)$ consensus rate. In Alice H. Oh, Alekh Agarwal,
Danielle Belgrave, and Kyunghyun Cho, editors,Advances in Neural
## Information Processing Systems, 2022.
[55]Konstantin Sozinov, Vladimir Vlassov, and Sarunas Girdzijauskas. Hu-
man activity recognition using federated learning. In2018 IEEE Intl
Conf on Parallel & Distributed Processing with Applications, Ubiquitous
## Computing & Communications, Big Data & Cloud Computing, Social
## Computing & Networking, Sustainable Computing & Communications
(ISPA/IUCC/BDCloud/SocialCom/SustainCom), pages 1103–1111. IEEE,
## 2018.
[56]Yuki Takezawa, Ryoma Sato, Han Bao, Kenta Niwa, and Makoto Ya-
mada. Beyond exponential graph: Communication-ecient topologies
for decentralized learning via nite-time convergence.Advances in
## Neural Information Processing Systems, 36, 2023.
[57]Yuki Takezawa and Sebastian U Stich. Scalable decentralized learning
with teleportation. InInternational Conference on Learning Representa-
tions (ICLR), 2025.
## [58]
## Yuming Tang, Yitian Zhang, Tao Niu, Zhen Li, Zijian Zhang, Huaping
Chen, and Long Zhang.  A survey on blockchain-based federated
learning: Categorization, application and analysis.CMES - Computer
Modeling in Engineering and Sciences, 139(3):2451–2477, 2024.
## [59]
Tribler Team. Ipv8 networking library.hps://github.com/tribler/py-
ipv8.
[60]Thijs Vogels, Hadrien Hendrikx, and Martin Jaggi. Beyond spectral
gap: the role of the topology in decentralized learning. In Alice H.
Oh, Alekh Agarwal, Danielle Belgrave, and Kyunghyun Cho, editors,
Advances in Neural Information Processing Systems, 2022.
## [61]
Leon Witt, Usama Zafar, KuoYeh Shen, Felix Sattler, Dan Li, Songtao
Wang, and Wojciech Samek. Decentralized and incentivized federated
## 10

Practical Federated Learning without a ServerEuroMLSys ’25, March 30-April 3 2025, Roerdam, Netherlands
learning: A blockchain-enabled framework utilising compressed soft-
labels and peer consistency.IEEE Transactions on Services Computing,
## 17(4):1449–1464, 2024.
[62]WonderNetwork. Global ping statistics.hps://wondernetwork.com/
pings. Accessed: 2022-05-12.
[63]Wentai Wu, Ligang He, Weiwei Lin, Rui Mao, Carsten Maple, and
Stephen Jarvis. Safa: A semi-asynchronous protocol for fast federated
learning with low overhead.IEEE Transactions on Computers, 70(5):655–
## 668, 2020.
[64]Bicheng Ying, Kun Yuan, Yiming Chen, Hanbin Hu, Pan Pan, and
Wotao Yin. Exponential graph is provably ecient for decentralized
deep training.Advances in Neural Information Processing Systems,
## 34:13975–13987, 2021.
[65]Liangqi Yuan, Ziran Wang, Lichao Sun, S Yu Philip, and Christopher G
Brinton. Decentralized federated learning: A survey and perspective.
IEEE Internet of Things Journal, 2024.
## [66]
## Feilong Zhang, Xianming Liu, Shiyi Lin, Gang Wu, Xiong Zhou, Jun-
jun Jiang, and Xiangyang Ji. No one idles: Ecient heterogeneous
federated learning with parallel edge and server computation. InInter-
national Conference on Machine Learning, pages 41399–41413. PMLR,
## 2023.
[67]Haoran Zhang, Shan Jiang, and Shichang Xuan. Decentralized fed-
erated learning based on blockchain: concepts, framework, and chal-
lenges.Computer Communications, 216:140–150, 2024.
[68]Hongyi Zhang, Jan Bosch, and Helena Holmström Olsson. Federated
learning systems: Architecture alternatives. In2020 27th Asia-Pacic
Software Engineering Conference (APSEC), pages 385–394. IEEE, 2020.
## 11