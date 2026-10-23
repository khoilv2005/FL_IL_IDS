"""Application message bytes, including setup. No raw-data transport API."""
from collections import Counter
from dataclasses import dataclass, field
import pickle


class CommunicationBudgetExceeded(RuntimeError):
    pass


@dataclass
class WireLedger:
    budget_bytes: int
    events: list = field(default_factory=list)
    bytes_sent: int = 0

    def transmit(self, sender, receiver, kind, message):
        if kind not in {'announcement', 'receiver_reference', 'head_reference', 'head_delta',
                        'router_sketches', 'verification_request', 'verification_receipt'}:
            raise ValueError('Unknown application message kind')
        if int(sender) == int(receiver):
            raise ValueError('Local computation is not a peer transmission')
        payload = pickle.dumps(message, protocol=5)
        count = len(payload)
        if self.bytes_sent + count > self.budget_bytes:
            raise CommunicationBudgetExceeded('Teaching wire budget exhausted before transmission')
        self.events.append(dict(sender=int(sender), receiver=int(receiver), kind=kind, bytes=count))
        self.bytes_sent += count
        # Deserializing here models an independent peer message, not shared mutable state.
        return pickle.loads(payload)

    def state_dict(self):
        return dict(budget_bytes=self.budget_bytes, events=list(self.events), bytes_sent=self.bytes_sent)

    @classmethod
    def from_state(cls, saved):
        ledger = cls(int(saved['budget_bytes']), list(saved['events']), int(saved['bytes_sent']))
        if sum(int(e['bytes']) for e in ledger.events) != ledger.bytes_sent or not 0 <= ledger.bytes_sent <= ledger.budget_bytes:
            raise ValueError('Invalid restored wire ledger')
        return ledger

    def summary(self):
        counts = Counter()
        for event in self.events:
            counts[event['kind']] += event['bytes']
        return dict(teaching_bytes=self.bytes_sent, by_kind=dict(counts), events=list(self.events),
                    native_bytes=None, total_system_bytes=None,
                    native_accounting='not measured by this ledger; do not claim total-system saving')
