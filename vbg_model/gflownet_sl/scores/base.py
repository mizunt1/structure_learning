import sys

from collections import namedtuple

LocalScore = namedtuple('LocalScore', ['key', 'score'])

class BaseScore:
    def __init__(self, data):
        self.data = data
        self.column_names = list(data.columns)

    def __call__(self, index, in_queue, out_queue, error_queue):
        try:
            while True:
                data = in_queue.get()
                if data is None:
                    break

                target, indices, indices_after = data
                local_score_before, local_score_after = self.get_local_scores(
                    target, indices, indices_after=indices_after)

                out_queue.put((True, *local_score_after))
                if local_score_before is not None:
                    out_queue.put((True, *local_score_before))

        except (KeyboardInterrupt, Exception):
            error_queue.put((index,) + sys.exc_info()[:2])
            out_queue.put((False, None, None))

    def get_local_scores(self, target, indices, indices_after=None):
        raise NotImplementedError('')
