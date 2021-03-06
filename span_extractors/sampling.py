import random
from abc import ABC, abstractmethod
from typing import List, Iterable, Dict

import time
import torch
from torch import multiprocessing

from span_extractors import util
from span_extractors.entities import Dataset

multiprocessing.set_sharing_strategy('file_system')


class TrainTensorBatch:
    def __init__(self, encodings: torch.tensor, ctx_masks: torch.tensor,
                 entity_masks: torch.tensor, entity_sizes: torch.tensor,
                 entity_sample_masks: torch.tensor, rels: torch.tensor, rel_masks: torch.tensor,
                 rel_sample_masks: torch.tensor, entity_types: torch.tensor, rel_types: torch.tensor,
                 entity_entries: List[List[Dict]], rel_entries: List[List[Dict]]):
        self.encodings = encodings
        self.ctx_masks = ctx_masks

        self.entity_masks = entity_masks
        self.entity_sizes = entity_sizes
        self.entity_types = entity_types
        self.entity_sample_masks = entity_sample_masks
        self.entity_entries = entity_entries

        self.rels = rels
        self.rel_masks = rel_masks
        self.rel_types = rel_types
        self.rel_sample_masks = rel_sample_masks
        self.rel_entries = rel_entries

    def to(self, device):
        encodings = self.encodings.to(device)
        ctx_masks = self.ctx_masks.to(device)

        entity_masks = self.entity_masks.to(device)
        entity_sizes = self.entity_sizes.to(device)
        entity_sample_masks = self.entity_sample_masks.to(device)

        rels = self.rels.to(device)
        rel_masks = self.rel_masks.to(device)
        rel_sample_masks = self.rel_sample_masks.to(device)

        entity_types = self.entity_types.to(device)
        rel_types = self.rel_types.to(device)

        entity_entries = self.entity_entries
        rel_entries = self.rel_entries

        return TrainTensorBatch(encodings, ctx_masks, entity_masks, entity_sizes, entity_sample_masks,
                                rels, rel_masks, rel_sample_masks, entity_types, rel_types, entity_entries,
                                rel_entries)


class EvalTensorBatch:
    def __init__(self, encodings: torch.tensor, ctx_masks: torch.tensor,
                 entity_masks: torch.tensor, entity_sizes: torch.tensor, entity_spans: torch.tensor,
                 entity_sample_masks: torch.tensor, entity_entries: List[List[Dict]]):
        self.encodings = encodings
        self.ctx_masks = ctx_masks

        self.entity_masks = entity_masks
        self.entity_sizes = entity_sizes
        self.entity_spans = entity_spans
        self.entity_sample_masks = entity_sample_masks
        self.entity_entries = entity_entries

    def to(self, device):
        encodings = self.encodings.to(device)
        ctx_masks = self.ctx_masks.to(device)

        entity_masks = self.entity_masks.to(device)
        entity_sizes = self.entity_sizes.to(device)
        entity_spans = self.entity_spans.to(device)
        entity_sample_masks = self.entity_sample_masks.to(device)
        entity_entries = self.entity_entries

        return EvalTensorBatch(encodings, ctx_masks, entity_masks, entity_sizes, entity_spans, 
                               entity_sample_masks, entity_entries)


class TrainTensorSample:
    def __init__(self, encoding: torch.tensor, ctx_mask: torch.tensor, entity_masks: torch.tensor,
                 entity_sizes: torch.tensor, rels: torch.tensor, rel_masks: torch.tensor,
                 entity_types: torch.tensor, rel_types: torch.tensor, entity_entries: List[Dict],
                 rel_entries: List[Dict]):
        self.encoding = encoding
        self.ctx_mask = ctx_mask

        self.entity_masks = entity_masks
        self.entity_sizes = entity_sizes
        self.entity_types = entity_types
        self.entity_entries = entity_entries

        self.rels = rels
        self.rel_masks = rel_masks
        self.rel_types = rel_types
        self.rel_entries = rel_entries


class EvalTensorSample:
    def __init__(self, encoding: torch.tensor, ctx_mask: torch.tensor, entity_masks: torch.tensor,
                 entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_entries: List[Dict]):
        self.encoding = encoding
        self.ctx_mask = ctx_mask

        self.entity_masks = entity_masks
        self.entity_sizes = entity_sizes
        self.entity_spans = entity_spans
        self.entity_entries = entity_entries


class Sampler:
    def __init__(self, processes: int, limit: int):
        # multiprocessing
        self._processes = processes
        self._limit = limit
        self._ctx = multiprocessing.get_context("spawn") if processes > 0 else None
        self._manager = self._ctx.Manager() if processes > 0 else None
        self._pool = self._ctx.Pool(processes=processes) if processes > 0 else None

    def create_train_sampler(self, dataset: Dataset, batch_size: int, max_span_size: int,
                             context_size: int, neg_entity_count: int, neg_rel_count: int,
                             order: Iterable = None, truncate: bool = False, relations_only = False,
                             type_key = "type"):
        train_sampler = TrainSampler(dataset, batch_size, max_span_size, context_size,
                                     neg_entity_count, neg_rel_count, order, truncate,
                                     self._manager, self._pool, self._processes, self._limit, relations_only,
                                     type_key)
        return train_sampler

    def create_eval_sampler(self, dataset: Dataset, batch_size: int, max_span_size: int, context_size: int,
                            order: Iterable = None, truncate: bool = False, relations_only = False,
                            type_key = "type"):
        eval_sampler = EvalSampler(dataset, batch_size, max_span_size, context_size,
                                   order, truncate, self._manager, self._pool, self._processes, self._limit, relations_only,
                                   type_key)
        return eval_sampler

    def join(self):
        if self._processes > 0:
            self._pool.close()
            self._pool.join()


class BaseSampler(ABC):
    def __init__(self, mp_func, manager, pool, processes, limit):
        # multiprocessing
        self._mp_func = mp_func
        self._manager = manager
        self._pool = pool
        self._processes = processes

        # avoid large memory consumption (e.g. in case of slow evaluation)
        self._semaphore = self._manager.Semaphore(limit) if processes > 0 else None

        self._current_batch = 0
        self._results = None

    @property
    @abstractmethod
    def _batches(self) -> List:
        pass

    def __next__(self):
        if self._current_batch < len(self._batches):
            if self._processes > 0:
                # multiprocessing
                batch, _ = self._results.next()
                self._semaphore.release()
            else:
                # no multiprocessing
                batch, _ = self._mp_func(self._batches[self._current_batch])

            self._current_batch += 1
            return batch
        else:
            raise StopIteration

    def __iter__(self):
        if self._processes > 0:
            # multiprocessing
            self._results = self._pool.imap(self._mp_func, self._batches)
        return self


class TrainSampler(BaseSampler):
    def __init__(self, dataset, batch_size, max_span_size, context_size, neg_entity_count, neg_rel_count,
                 order, truncate, manager, pool, processes, limit, relations_only, type_key):
        super().__init__(_produce_train_batch, manager, pool, processes, limit)

        self._dataset = dataset
        self._batch_size = batch_size
        self._max_span_size = max_span_size
        self._context_size = context_size

        self._neg_entity_count = neg_entity_count
        self._neg_rel_count = neg_rel_count
        self._relations_only = relations_only
        self._type_key = type_key

        batches = self._dataset.iterate_documents(self._batch_size, order=order, truncate=truncate)
        self._prep_batches = self._prepare(batches)

    def _prepare(self, batches):
        prep_batches = []
        for i, batch in enumerate(batches):
            prep_batches.append((i, batch, self._neg_entity_count, self._neg_rel_count,
                                 self._max_span_size, self._context_size, self._semaphore, 
                                 self._relations_only, self._type_key))

        return prep_batches

    @property
    def _batches(self):
        return self._prep_batches


class EvalSampler(BaseSampler):
    def __init__(self, dataset, batch_size, max_span_size, context_size,
                 order, truncate, manager, pool, processes, limit, relations_only, type_key):
        super().__init__(_produce_eval_batch, manager, pool, processes, limit)

        self._dataset = dataset
        self._batch_size = batch_size
        self._max_span_size = max_span_size
        self._context_size = context_size
        self._relations_only = relations_only
        self._type_key = type_key

        batches = self._dataset.iterate_documents(self._batch_size, order=order, truncate=truncate)
        self._prep_batches = self._prepare(batches)

    def _prepare(self, batches):
        prep_batches = []

        for i, batch in enumerate(batches):
            prep_batches.append((i, batch, self._max_span_size, self._context_size, self._semaphore, 
                                 self._relations_only, self._type_key))

        return prep_batches

    @property
    def _batches(self):
        return self._prep_batches


def _produce_train_batch(args):
    i, docs, neg_entity_count, neg_rel_count, max_span_size, context_size, semaphore, relations_only, type_key = args

    if semaphore is not None:
        semaphore.acquire()

    samples = []
    for d in docs:
        sample = _create_train_sample(d, neg_entity_count, neg_rel_count, max_span_size, context_size, 
                                      relations_only=relations_only, type_key=type_key)
        samples.append(sample)

    batch = _create_train_batch(samples)

    return batch, i


def _produce_eval_batch(args):
    i, docs, max_span_size, context_size, semaphore, relations_only, type_key = args

    if semaphore is not None:
        semaphore.acquire()

    samples = []
    for d in docs:
        sample = _create_eval_sample(d, max_span_size, context_size, relations_only=relations_only, type_key=type_key)
        samples.append(sample)

    batch = _create_eval_batch(samples)
    return batch, i


def _create_train_sample(doc, neg_entity_count, neg_rel_count, max_span_size, context_size, type_key="type", relations_only=False):
    encoding = doc.encoding
    token_count = len(doc.tokens)

    # positive entities
    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes, pos_entity_entries = [], [], [], [], []
    for e in doc.entities:
        pos_entity_spans.append(e.span)
        pos_entity_types.append(e.entity_type.index)
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))
        pos_entity_sizes.append(len(e.tokens))
        pos_entity_entries.append({"type_string": e.entity_type.verbose_name, 
                                   "phrase": e.phrase, 
                                   type_key: e.entity_type.index,
                                   "indicator": "[ENT]"})
        
    # positive relations
    pos_rels, pos_rel_spans, pos_rel_types, pos_rel_masks, pos_rel_entries = [], [], [], [], []
    for rel in doc.relations:
        s1, s2 = rel.head_entity.span, rel.tail_entity.span
        pos_rels.append((pos_entity_spans.index(s1), pos_entity_spans.index(s2)))
        pos_rel_spans.append((s1, s2))
        pos_rel_types.append(rel.relation_type) # convert to type index later
        pos_rel_masks.append(create_rel_mask(s1, s2, context_size))
        phrase = "|{}| {} |{}|".format(rel.head_entity.phrase, rel.relation_type.verbose_name, rel.tail_entity.phrase)
        pos_rel_entries.append({"type_string": rel.relation_type.verbose_name, 
                                "phrase": phrase, 
                                type_key: rel.relation_type.index,
                                "indicator": "[REL]"})

    # negative entities
    neg_entity_spans, neg_entity_sizes, neg_entity_masks, neg_entity_types, neg_entity_entries = [], [], [], [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = doc.tokens[i:i + size].span
            phrase = doc.tokens[i:i + size].span_phrase
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)
                neg_entity_masks.append(create_entity_mask(*span, context_size))
                neg_entity_types.append(0) # 0 = none type
                neg_entity_entries.append({"type_string": "O", 
                                           "phrase": phrase, 
                                           type_key: 0,
                                           "indicator": "[ENT]"})

    # sample negative entities
    neg_entity_samples = random.sample(list(zip(neg_entity_spans, neg_entity_sizes, neg_entity_masks, 
                                       neg_entity_types, neg_entity_entries)), min(len(neg_entity_spans), neg_entity_count))
    neg_entity_spans, neg_entity_sizes, neg_entity_masks, neg_entity_types, neg_entity_entries = zip(
        *neg_entity_samples) if neg_entity_samples else ([], [], [], [], []) 

    # neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    # neg_entity_types = [0] * len(neg_entity_spans)

    # negative relations
    # use only strong negative relations, i.e. pairs of actual (labeled) entities that are not related
    neg_rel_spans, neg_rel_masks, neg_rel_types, neg_rel_entries = [], [], [], []

    for i, s1 in enumerate(pos_entity_spans):
        for j, s2 in enumerate(pos_entity_spans):
            rev = (s2, s1)
            rev_symmetric = rev in pos_rel_spans and pos_rel_types[pos_rel_spans.index(rev)].symmetric

            # do not add as negative relation sample:
            # neg. relations from an entity to itself
            # entity pairs that are related according to gt
            # entity pairs whose reverse exists as a symmetric relation in gt
            if s1 != s2 and (s1, s2) not in pos_rel_spans and not rev_symmetric:
                spans = (s1, s2)
                neg_rel_spans.append(spans)
                neg_rel_masks.append(create_rel_mask(*spans, context_size))
                neg_rel_types.append(0)
                phrase = "|{}| {} |{}|".format(pos_entity_entries[i]["phrase"], "O", pos_entity_entries[j]["phrase"])
                neg_rel_entries.append({"type_string": "O", 
                                        "phrase": phrase, 
                                        type_key: 0,
                                        "indicator": "[REL]"})


    # sample negative relations
    # neg_rel_spans = random.sample(neg_rel_spans, min(len(neg_rel_spans), neg_rel_count))
    neg_rel_samples = random.sample(list(zip(neg_rel_spans, neg_rel_masks, 
                                    neg_rel_types, neg_rel_entries)), min(len(neg_rel_spans), neg_rel_count))
    neg_rel_spans, neg_rel_masks, neg_rel_types, neg_rel_entries = zip(
        *neg_rel_samples) if neg_rel_samples else ([], [], [], []) 
    neg_rels = [(pos_entity_spans.index(s1), pos_entity_spans.index(s2)) for s1, s2 in neg_rel_spans]
    # neg_rel_masks = [create_rel_mask(*spans, context_size) for spans in neg_rel_spans]
    # neg_rel_types = [0] * len(neg_rel_spans)

    # merge
    entity_types = pos_entity_types + list(neg_entity_types) if not relations_only else pos_entity_types
    entity_masks = pos_entity_masks + list(neg_entity_masks) if not relations_only else pos_entity_masks
    entity_sizes = pos_entity_sizes + list(neg_entity_sizes) if not relations_only else pos_entity_sizes
    entity_entries = pos_entity_entries + list(neg_entity_entries) if not relations_only else pos_entity_entries

    rels = pos_rels + neg_rels
    rel_types = [r.index for r in pos_rel_types] + list(neg_rel_types)
    rel_masks = pos_rel_masks + list(neg_rel_masks)
    rel_entries = pos_rel_entries + list(neg_rel_entries)

    assert len(entity_masks) == len(entity_sizes) == len(entity_types) == len(entity_entries)
    assert len(rels) == len(rel_masks) == len(rel_types) == len(rel_entries)

    # create tensors
    # token indices
    _encoding = encoding
    encoding = torch.zeros(context_size, dtype=torch.long)
    encoding[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    ctx_mask = torch.zeros(context_size, dtype=torch.bool)
    ctx_mask[:len(_encoding)] = 1

    # entities
    entity_masks = torch.stack(entity_masks)
    entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
    entity_types = torch.tensor(entity_types, dtype=torch.long)

    # relations
    rels = torch.tensor(rels, dtype=torch.long) if rels else torch.zeros([0, 2], dtype=torch.long)
    rel_masks = torch.stack(rel_masks) if rel_masks else torch.zeros([0, context_size], dtype=torch.bool)
    rel_types = torch.tensor(rel_types, dtype=torch.long) if rel_types else torch.zeros([0], dtype=torch.long)

    return TrainTensorSample(encoding=encoding, ctx_mask=ctx_mask, entity_masks=entity_masks,
                             entity_sizes=entity_sizes, entity_types=entity_types,
                             rels=rels, rel_masks=rel_masks, rel_types=rel_types, 
                             entity_entries=entity_entries, rel_entries=rel_entries)


def _create_eval_sample(doc, max_span_size, context_size, type_key="type_index", relations_only=False):
    encoding = doc.encoding
    token_count = len(doc.tokens)

    # create entity candidates
    entity_spans = []
    entity_masks = []
    entity_sizes = []
    entity_entries = []

    if relations_only:
        for entity in doc.entities:
            span = entity.span
            phrase = doc.tokens[span[0]:span[1]]
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_sizes.append(span[1]-span[0])
            entity_entries.append({"phrase": phrase,
                                   "type_string": entity.entity_type.verbose_name,
                                    type_key: entity.entity_type.index})
    else:
        for size in range(1, max_span_size + 1):
            for i in range(0, (token_count - size) + 1):
                span = doc.tokens[i:i + size].span
                phrase = doc.tokens[i:i + size].span_phrase
                entity_spans.append(span)
                entity_masks.append(create_entity_mask(*span, context_size))
                entity_sizes.append(size)
                entity_entries.append({"phrase": phrase,
                                    "type_string": "<TBD>",
                                    type_key: "-1"})

    # create tensors
    # token indices
    _encoding = encoding
    encoding = torch.zeros(context_size, dtype=torch.long)
    encoding[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    ctx_mask = torch.zeros(context_size, dtype=torch.bool)
    ctx_mask[:len(_encoding)] = 1

    # entities
    entity_masks = torch.stack(entity_masks) if entity_masks else torch.zeros([1, context_size], dtype=torch.bool)
    entity_sizes = torch.tensor(entity_sizes, dtype=torch.long) if entity_sizes else torch.tensor([0], dtype=torch.long)
    entity_spans = torch.tensor(entity_spans, dtype=torch.long) if  entity_spans else torch.tensor([(0,0)], dtype=torch.long)

    return EvalTensorSample(encoding=encoding, ctx_mask=ctx_mask, entity_masks=entity_masks,
                            entity_sizes=entity_sizes, entity_spans=entity_spans, 
                            entity_entries=entity_entries)


def _create_train_batch(samples):
    batch_encodings = []
    batch_ctx_masks = []

    batch_entity_masks = []
    batch_entity_sizes = []
    batch_entity_sample_masks = []

    batch_rels = []
    batch_rel_sample_masks = []
    batch_rel_masks = []

    batch_entity_types = []
    batch_rel_types = []

    batch_entity_entries = []
    batch_rel_entries = []

    for sample in samples:
        encoding = sample.encoding
        ctx_mask = sample.ctx_mask

        # entities
        entity_masks = sample.entity_masks
        entity_sizes = sample.entity_sizes
        entity_types = sample.entity_types

        # relations
        rels = sample.rels
        rel_masks = sample.rel_masks
        rel_types = sample.rel_types

        # tensors to mask entity/relation samples of batch
        # since samples are stacked into batches, "padding" entities/relations possibly must be created
        # these are later masked during loss computation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
        rel_sample_masks = torch.tensor([1] * rel_masks.shape[0], dtype=torch.bool)

        # corner case handling (no entities / relations)
        if entity_masks.shape[0] == 0:
            entity_types = torch.tensor([0], dtype=torch.long)
            entity_masks = torch.tensor([[0] * entity_masks.shape[-1]], dtype=torch.bool)
            entity_sizes = torch.tensor([0], dtype=torch.long)
            entity_sample_masks = torch.tensor([0], dtype=torch.bool)

        if rel_masks.shape[0] == 0:
            rels = torch.tensor([[0, 0]], dtype=torch.long)
            rel_types = torch.tensor([0], dtype=torch.long)
            rel_masks = torch.tensor([[0] * rel_masks.shape[-1]], dtype=torch.bool)
            rel_sample_masks = torch.tensor([0], dtype=torch.bool)

        batch_encodings.append(encoding)
        batch_ctx_masks.append(ctx_mask)

        batch_entity_masks.append(entity_masks)
        batch_entity_sizes.append(entity_sizes)
        batch_entity_sample_masks.append(entity_sample_masks)

        batch_rels.append(rels)
        batch_rel_masks.append(rel_masks)
        batch_rel_sample_masks.append(rel_sample_masks)

        batch_rel_types.append(rel_types)
        batch_entity_types.append(entity_types)

        batch_entity_entries.append(sample.entity_entries)
        batch_rel_entries.append(sample.rel_entries)

    # stack samples
    encodings = util.padded_stack(batch_encodings)
    ctx_masks = util.padded_stack(batch_ctx_masks)

    batch_entity_masks = util.padded_stack(batch_entity_masks)
    batch_entity_sizes = util.padded_stack(batch_entity_sizes)

    batch_rels = util.padded_stack(batch_rels)
    batch_rel_masks = util.padded_stack(batch_rel_masks)

    batch_rel_sample_masks = util.padded_stack(batch_rel_sample_masks)
    batch_entity_sample_masks = util.padded_stack(batch_entity_sample_masks)

    batch_rel_types = util.padded_stack(batch_rel_types)
    batch_entity_types = util.padded_stack(batch_entity_types)

    batch_entity_entries = util.padded_entries(batch_entity_entries)
    batch_rel_entries = util.padded_entries(batch_rel_entries)

    batch = TrainTensorBatch(encodings=encodings, ctx_masks=ctx_masks, entity_masks=batch_entity_masks,
                             entity_sizes=batch_entity_sizes,
                             entity_types=batch_entity_types,
                             entity_sample_masks=batch_entity_sample_masks, rels=batch_rels, rel_masks=batch_rel_masks,
                             rel_types=batch_rel_types, rel_sample_masks=batch_rel_sample_masks,
                             entity_entries=batch_entity_entries, rel_entries=batch_rel_entries)

    return batch


def _create_eval_batch(samples):
    batch_encodings = []
    batch_ctx_masks = []

    batch_entity_masks = []
    batch_entity_sizes = []
    batch_entity_spans = []
    batch_entity_sample_masks = []
    batch_entity_entries = []

    for sample in samples:
        encoding = sample.encoding
        ctx_mask = sample.ctx_mask

        entity_masks = sample.entity_masks
        entity_sizes = sample.entity_sizes
        entity_spans = sample.entity_spans

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)

        # corner case handling (no entities)
        if entity_masks.shape[0] == 0:
            entity_masks = torch.tensor([[0] * entity_masks.shape[-1]], dtype=torch.bool)
            entity_sizes = torch.tensor([0], dtype=torch.long)
            entity_spans = torch.tensor([[0, 0]], dtype=torch.long)
            entity_sample_masks = torch.tensor([0], dtype=torch.bool)

        batch_encodings.append(encoding)
        batch_ctx_masks.append(ctx_mask)

        batch_entity_masks.append(entity_masks)
        batch_entity_sizes.append(entity_sizes)
        batch_entity_spans.append(entity_spans)
        batch_entity_sample_masks.append(entity_sample_masks)

        batch_entity_entries.append(sample.entity_entries)

    # stack samples
    encodings = util.padded_stack(batch_encodings)
    ctx_masks = util.padded_stack(batch_ctx_masks)

    batch_entity_masks = util.padded_stack(batch_entity_masks)
    batch_entity_sizes = util.padded_stack(batch_entity_sizes)
    batch_entity_spans = util.padded_stack(batch_entity_spans)
    batch_entity_sample_masks = util.padded_stack(batch_entity_sample_masks)
    batch_entity_entries = util.padded_entries(batch_entity_entries)

    batch = EvalTensorBatch(encodings=encodings, ctx_masks=ctx_masks, entity_masks=batch_entity_masks,
                            entity_sizes=batch_entity_sizes, entity_spans=batch_entity_spans,
                            entity_sample_masks=batch_entity_sample_masks,
                            entity_entries=batch_entity_entries)

    return batch


def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask


def create_rel_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask
