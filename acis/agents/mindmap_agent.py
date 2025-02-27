"""
Mind Map Agent

This module implements the Mind Map agent for building knowledge graphs
of competitive intelligence data.
"""

import asyncio
import json
import logging
import uuid
from typing import List, Dict, Any, Optional

from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

from acis.agents.search_agent import SearchAgent
from acis.config.settings import settings

# Configure logging
logger = logging.getLogger(__name__)


class MindMapAgent:
    """Agent for generating mind maps and knowledge graphs of competitive intelligence."""

    def __init__(self):
        """Initialize the mind map agent."""
        self.llm = OpenAI(
            temperature=0.3,
            model_name=settings.llm_model,
            openai_api_key=settings.openai_api_key
        )
        
        # Initialize search agent for data gathering
        self.search_agent = SearchAgent()
        
        # Define node generation prompt
        self.node_prompt = PromptTemplate(
            input_variables=["entity", "relation"],
            template="""
            Generate a list of {relation} nodes related to {entity}.
            
            For each node, provide:
            1. A label (short name)
            2. A type (e.g., product, event, financial, strategy)
            3. A brief description
            
            Format your response as a JSON array of objects with the following structure:
            [
              {{
                "label": "Node label",
                "type": "Node type",
                "description": "Brief description"
              }}
            ]
            
            Focus on recent and significant aspects of {relation} for {entity}.
            Only provide the JSON array, no other text.
            """
        )
        
        self.node_chain = LLMChain(llm=self.llm, prompt=self.node_prompt)
        
        # Define edge generation prompt
        self.edge_prompt = PromptTemplate(
            input_variables=["entity", "nodes"],
            template="""
            For the company {entity} and the following nodes:
            
            {nodes}
            
            Generate a list of relationships (edges) between {entity} and these nodes, or between the nodes themselves.
            
            For each relationship, provide:
            1. Source node (either "{entity}" or one of the node labels)
            2. Target node (one of the node labels, not "{entity}")
            3. Relationship type (verb phrase describing the relationship)
            4. A brief explanation of the relationship
            
            Format your response as a JSON array of objects with the following structure:
            [
              {{
                "source": "Source node",
                "target": "Target node",
                "type": "Relationship type",
                "explanation": "Brief explanation"
              }}
            ]
            
            Only provide the JSON array, no other text.
            """
        )
        
        self.edge_chain = LLMChain(llm=self.llm, prompt=self.edge_prompt)

    async def generate_mindmap(self, entity: str, relations: List[str]) -> Dict[str, Any]:
        """
        Generate a mind map for a given entity and set of relations.
        
        Args:
            entity: The main entity (e.g., company name)
            relations: List of relation types to explore
            
        Returns:
            Mind map as a graph structure with nodes and edges
        """
        try:
            # Create the entity node
            nodes = [
                {
                    "id": str(uuid.uuid4()),
                    "label": entity,
                    "type": "company",
                    "data": {
                        "description": f"Main entity: {entity}"
                    }
                }
            ]
            
            # Track node labels to IDs for edge creation
            node_map = {entity: nodes[0]["id"]}
            
            # Generate nodes for each relation
            for relation in relations:
                relation_nodes = await self._generate_nodes(entity, relation)
                
                # Add generated nodes to the overall node list
                for node in relation_nodes:
                    node_id = str(uuid.uuid4())
                    node_map[node["label"]] = node_id
                    
                    nodes.append({
                        "id": node_id,
                        "label": node["label"],
                        "type": node["type"],
                        "data": {
                            "description": node.get("description", ""),
                            "relation": relation
                        }
                    })
            
            # Generate edges between nodes
            edges = await self._generate_edges(entity, nodes, node_map)
            
            # Return the complete graph
            return {
                "nodes": nodes,
                "edges": edges
            }
        except Exception as e:
            logger.error(f"Error generating mind map: {e}")
            # Return a minimal valid graph
            return {
                "nodes": [
                    {
                        "id": "1",
                        "label": entity,
                        "type": "company",
                        "data": {}
                    }
                ],
                "edges": []
            }

    async def _generate_nodes(self, entity: str, relation: str) -> List[Dict[str, Any]]:
        """
        Generate nodes for a specific relation.
        
        Args:
            entity: The main entity
            relation: The relation type
            
        Returns:
            List of nodes
        """
        try:
            # Use LLM to generate nodes
            response = await asyncio.to_thread(
                self.node_chain.run,
                entity=entity,
                relation=relation
            )
            
            # Parse JSON response
            try:
                nodes = json.loads(response)
                logger.info(f"Generated {len(nodes)} nodes for {entity} - {relation}")
                return nodes
            except json.JSONDecodeError:
                logger.error(f"Failed to parse node JSON: {response}")
                # Return a fallback node if parsing fails
                return [
                    {
                        "label": f"{relation} for {entity}",
                        "type": "generic",
                        "description": f"A {relation} related to {entity}"
                    }
                ]
        except Exception as e:
            logger.error(f"Error generating nodes: {e}")
            return []

    async def _generate_edges(self, entity: str, nodes: List[Dict[str, Any]], 
                             node_map: Dict[str, str]) -> List[Dict[str, Any]]:
        """
        Generate edges between nodes.
        
        Args:
            entity: The main entity
            nodes: List of all nodes
            node_map: Mapping from node labels to IDs
            
        Returns:
            List of edges
        """
        try:
            # Format nodes for the prompt
            node_descriptions = []
            for node in nodes:
                if node["label"] != entity:  # Skip the main entity node
                    desc = f"- {node['label']} (Type: {node['type']})"
                    if "description" in node.get("data", {}):
                        desc += f": {node['data']['description']}"
                    node_descriptions.append(desc)
            
            nodes_text = "\n".join(node_descriptions)
            
            # Use LLM to generate edges
            response = await asyncio.to_thread(
                self.edge_chain.run,
                entity=entity,
                nodes=nodes_text
            )
            
            # Parse JSON response
            try:
                raw_edges = json.loads(response)
                logger.info(f"Generated {len(raw_edges)} edges for {entity}")
                
                # Convert labels to IDs
                edges = []
                for edge in raw_edges:
                    source = edge.get("source")
                    target = edge.get("target")
                    
                    # Skip invalid edges
                    if not source or not target:
                        continue
                    
                    # Map labels to IDs
                    source_id = node_map.get(source)
                    target_id = node_map.get(target)
                    
                    # Skip if we can't find the nodes
                    if not source_id or not target_id:
                        continue
                    
                    edges.append({
                        "source": source_id,
                        "target": target_id,
                        "label": edge.get("type", "related"),
                        "type": edge.get("type", "related")
                    })
                
                return edges
            except json.JSONDecodeError:
                logger.error(f"Failed to parse edge JSON: {response}")
                return []
        except Exception as e:
            logger.error(f"Error generating edges: {e}")
            return [] 