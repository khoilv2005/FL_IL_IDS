"""Simulated P2P application-wire accounting; no claims about socket/TLS bytes."""
import hashlib
import json
from .config import Rejected


class Transport:
    def __init__(self,path,edges,protocol):
        self.path=path;self.edges={tuple(map(int,e)) for e in edges};self.protocol=protocol
        self.sent={};self.received={};self.records=[]

    def send(self,sender,receiver,kind,value):
        if (sender,receiver) not in self.edges:raise Rejected('UNAUTHORIZED_EDGE')
        data=value if isinstance(value,bytes) else json.dumps(value,sort_keys=True,separators=(',',':')).encode()
        n=len(data)
        if self.sent.get(sender,0)+n>self.protocol.max_outgoing_bytes:raise Rejected('OUTGOING_QUOTA')
        if self.received.get(receiver,0)+n>self.protocol.max_incoming_bytes:raise Rejected('INCOMING_QUOTA')
        self.sent[sender]=self.sent.get(sender,0)+n;self.received[receiver]=self.received.get(receiver,0)+n
        record=dict(sender=sender,receiver=receiver,kind=kind,bytes=n,sha256=hashlib.sha256(data).hexdigest())
        self.records.append(record)
        with self.path.open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
        return data

    def summary(self):
        return dict(application_egress_bytes=sum(self.sent.values()),sent=self.sent,received=self.received,
            retained_base_exchange_measured=False,limitation='Pairwise simulated application payload only; no network, security or full-round baseline-byte claim')
