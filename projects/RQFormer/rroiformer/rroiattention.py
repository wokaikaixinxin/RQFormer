from mmengine.model import  xavier_init
from mmengine.model import BaseModule, constant_init
from torch import Tensor, nn

from mmrotate.registry import MODELS
from mmdet.utils import OptConfigType


@MODELS.register_module()
class RRoIAttention(BaseModule):
    """Implements RRoIAttention.

    Args:
        embed_dims (int): The embedding dimensions of query.
            Defaults to 256.
        roi_pooler_resolution (int): The shape of roi feature.
            Defaults to 7.
        act_cfg (dict): The activation config for RRoIAttention.
        norm_cfg (dict): Config dict for normalization layer. Default
            layer normalization.
        init_cfg (obj:`mmengine.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 roi_pooler_resolution: int = 7,
                 init_cfg: OptConfigType = None) -> None:
        super(RRoIAttention, self).__init__(init_cfg)
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.roi_pooler_resolution = roi_pooler_resolution
        self.attention_weights = nn.Linear(embed_dims, num_heads * (roi_pooler_resolution**2))
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()


    def init_weights(self) -> None:
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)


    def forward(self, query: Tensor, roi_feat: Tensor) -> Tensor:
        """Forward function for `RRoIAttention`.

        Args:
            query (Tensor): The feature can be used
                to generate the parameter, has shape
                (bs, num_queries, embed_dims).
            roi_feat (Tensor): Feature that
                interact with parameters, has shape
                (bs * num_queries, embed_dims, pooling_h , pooling_w).

        Returns:
            Tensor: The output feature has shape
            (bs, num_queries, embed_dims).
        """
        bs, num_queries = query.shape[:2]
        attention_weights = self.attention_weights(query).view(
            bs, num_queries, self.num_heads, self.roi_pooler_resolution**2)# bs, num_query, num_heads, pooling_h*pooling_w
        attention_weights = attention_weights.softmax(-1)
        attention_weights = attention_weights.unsqueeze(-2) # bs, num_query, num_heads, 1, pooling_h*pooling_w
        value = self.value_proj(roi_feat.permute(0, 2, 3, 1)).permute(0, 3, 1, 2).contiguous()
        value = value.view(bs, num_queries, self.num_heads, -1,
                              self.roi_pooler_resolution**2) # bs, num_query, num_heads, embed_dims//num_heads, pooling_h*pooling_w
        output = (value * attention_weights).sum(-1).view(
            bs, num_queries, self.embed_dims)                # bs, num_query, embed_dims

        output = self.output_proj(output)

        return output