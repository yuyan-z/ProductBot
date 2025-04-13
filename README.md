# ProductBot

# Graph
graph TD;
	__start__([<p>__start__</p>]):::first
	product_agent(product_agent)
	review_agent(review_agent)
	merge(merge)
	__end__([<p>__end__</p>]):::last
	__start__ --> product_agent;
	merge --> __end__;
	product_agent --> review_agent;
	review_agent --> merge;
	classDef default fill:#f2f0ff,line-height:1.2
	classDef first fill-opacity:0
	classDef last fill:#bfb6fc