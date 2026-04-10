from copy import deepcopy
from typing import List, Dict

from datasets import Dataset
from wtpsplit import SaT


class SaTSegmentation:
    def __init__(self, model: str, pattern='<seg>'):
        """
        e.g.: model = 'sat-12l-sm'
        """
        self.sat = SaT(model)
        self.pattern = pattern

    def _map_segment(self, x, pattern):
        if x == '':
            return '\n'
        else:
            return x + pattern
        
    def text_segmentation(self, text: str | List[str], pattern=None) -> str | List[str]:
        if not pattern:
            pattern = self.pattern

        if type(text) == str:
            text = [text]

        segments = self.sat.split(text)
        new_texts = []
        for segment in segments:
            segment = list(map(lambda x: self._map_segment(x, pattern), segment))
            new_text = ''.join(segment)
            new_texts.append(new_text)
        return new_texts if len(new_texts) > 1 else new_texts[0]

    def messages_segmentation(self, messages: List[Dict], pattern=None) -> List[Dict]:
        """
        messages input format:
        [
            {'role': ..., 'content': ...},
            ...
        ]
        """
        if not pattern:
            pattern = self.pattern

        messages = deepcopy(messages)

        contents = []
        for message in messages:
            contents.append(message['content'])

        segments = self.sat.split(contents)

        for message, segment in zip(messages, segments):
            segment = list(map(lambda x: self._map_segment(x, pattern), segment))
            message['content'] = ''.join(segment)

        return messages
    
    def dataset_segmentation(self, dataset: Dataset, keys: str | List[str], pattern=None) -> Dataset:
        """
        segment HF Dataset. please provide keys to specify the column to segment
        """
        if not pattern:
            pattern = self.pattern

        if type(keys) != list:
            keys = [keys]

        columns = {k: dataset[k] for k in keys}
        contents = []
        for _, content in columns.items():
            contents.extend(content)
        
        segments = self.sat.split(contents)

        i = 0
        for key in columns:
            new_contents = []
            for j in range(i, i + len(columns[key])):
                segment = list(map(lambda x: self._map_segment(x, pattern), segments[j]))
                new_contents.append(''.join(segment))
            i += len(columns[key])
            dataset = dataset.remove_columns(key)
            dataset = dataset.add_column(key, new_contents)

        return dataset