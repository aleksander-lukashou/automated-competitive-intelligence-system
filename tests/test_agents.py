"""
Tests for ACIS agent components.

This module contains unit and integration tests for the agent components.
"""

import pytest
import asyncio
import json
from unittest.mock import patch, MagicMock

from acis.agents.search_agent import SearchAgent
from acis.agents.mindmap_agent import MindMapAgent
from acis.agents.coding_agent import CodingAgent


class TestSearchAgent:
    """Tests for the SearchAgent class."""

    @pytest.fixture
    def agent(self):
        """Create a SearchAgent instance for testing."""
        with patch('langchain.llms.OpenAI') as mock_openai:
            # Configure the mock
            mock_chain = MagicMock()
            mock_chain.run.return_value = "Optimized test query"
            mock_openai.return_value.run = mock_chain.run
            
            agent = SearchAgent()
            
            # Replace the chains with mocks
            agent.search_chain = mock_chain
            agent.summarization_chain = mock_chain
            agent.sentiment_chain = mock_chain
            
            yield agent
    
    @pytest.mark.asyncio
    async def test_optimize_query(self, agent):
        """Test query optimization."""
        optimized = await agent._optimize_query("test query")
        assert isinstance(optimized, str)
        assert len(optimized) > 0
    
    @pytest.mark.asyncio
    async def test_perform_search(self, agent):
        """Test search functionality."""
        results = await agent._perform_search("test query", 5)
        assert isinstance(results, list)
        assert len(results) <= 5
        
        if results:
            assert "headline" in results[0]
            assert "url" in results[0]
    
    @pytest.mark.asyncio
    async def test_process_results(self, agent):
        """Test result processing."""
        test_results = [
            {
                "source": "Test Source",
                "headline": "Test Headline",
                "url": "https://example.com",
                "content": "Test content for processing."
            }
        ]
        
        with patch.object(agent.summarization_chain, 'run', return_value="Test summary"):
            with patch.object(agent.sentiment_chain, 'run', return_value="0.5"):
                processed = await agent._process_results(test_results)
                
                assert isinstance(processed, list)
                assert len(processed) == 1
                assert "summary" in processed[0]
                assert "sentiment" in processed[0]
                assert processed[0]["sentiment"] == 0.5
    
    @pytest.mark.asyncio
    async def test_search(self, agent):
        """Test the main search method."""
        with patch.object(agent, '_optimize_query', return_value="optimized query"):
            with patch.object(agent, '_perform_search', return_value=[{"source": "Test", "headline": "Test"}]):
                with patch.object(agent, '_process_results', return_value=[{"source": "Test", "headline": "Test", "summary": "Test"}]):
                    results = await agent.search("test", 5)
                    
                    assert isinstance(results, list)
                    assert len(results) > 0
                    assert "summary" in results[0]


class TestMindMapAgent:
    """Tests for the MindMapAgent class."""

    @pytest.fixture
    def agent(self):
        """Create a MindMapAgent instance for testing."""
        with patch('langchain.llms.OpenAI') as mock_openai:
            # Configure the mock
            mock_chain = MagicMock()
            mock_chain.run.return_value = '[{"label": "Test Node", "type": "product", "description": "A test node"}]'
            mock_openai.return_value.run = mock_chain.run
            
            agent = MindMapAgent()
            
            # Replace the chains with mocks
            agent.node_chain = mock_chain
            agent.edge_chain = mock_chain
            
            # Mock the search agent
            agent.search_agent = MagicMock()
            
            yield agent
    
    @pytest.mark.asyncio
    async def test_generate_nodes(self, agent):
        """Test node generation."""
        nodes = await agent._generate_nodes("Test Company", "Products")
        
        assert isinstance(nodes, list)
        assert len(nodes) > 0
        assert "label" in nodes[0]
        assert "type" in nodes[0]
    
    @pytest.mark.asyncio
    async def test_generate_edges(self, agent):
        """Test edge generation."""
        # Setup test data
        entity = "Test Company"
        nodes = [
            {"id": "1", "label": entity, "type": "company", "data": {}},
            {"id": "2", "label": "Product 1", "type": "product", "data": {}}
        ]
        node_map = {
            entity: "1",
            "Product 1": "2"
        }
        
        # Set up mock to return valid JSON for edges
        agent.edge_chain.run.return_value = '[{"source": "Test Company", "target": "Product 1", "type": "produces"}]'
        
        edges = await agent._generate_edges(entity, nodes, node_map)
        
        assert isinstance(edges, list)
        assert len(edges) > 0
        assert "source" in edges[0]
        assert "target" in edges[0]
    
    @pytest.mark.asyncio
    async def test_generate_mindmap(self, agent):
        """Test the main mindmap generation method."""
        # Set up node generation mock
        agent._generate_nodes = AsyncMock(return_value=[
            {"label": "Product 1", "type": "product", "description": "A test product"}
        ])
        
        # Set up edge generation mock
        agent._generate_edges = AsyncMock(return_value=[
            {"source": "1", "target": "2", "label": "produces", "type": "production"}
        ])
        
        graph = await agent.generate_mindmap("Test Company", ["Products"])
        
        assert isinstance(graph, dict)
        assert "nodes" in graph
        assert "edges" in graph
        assert len(graph["nodes"]) > 0
        assert isinstance(graph["edges"], list)


class TestCodingAgent:
    """Tests for the CodingAgent class."""

    @pytest.fixture
    def agent(self):
        """Create a CodingAgent instance for testing."""
        with patch('langchain.llms.OpenAI') as mock_openai:
            # Configure the mock
            mock_chain = MagicMock()
            mock_chain.run.return_value = "print('Hello, world!')\nresult = {'summary': 'Test summary'}"
            mock_openai.return_value.run = mock_chain.run
            
            agent = CodingAgent()
            
            # Replace the chains with mocks
            agent.code_chain = mock_chain
            agent.improvement_chain = mock_chain
            
            yield agent
    
    @pytest.mark.asyncio
    async def test_generate_code(self, agent):
        """Test code generation."""
        code = await agent._generate_code(
            task="Print hello world",
            data_description="No data needed",
            output_format="JSON with summary"
        )
        
        assert isinstance(code, str)
        assert len(code) > 0
    
    @pytest.mark.asyncio
    async def test_improve_code(self, agent):
        """Test code improvement."""
        improved = await agent._improve_code(
            code="print('Hello')",
            error="Missing result variable",
            improvement_goal="Add result variable"
        )
        
        assert isinstance(improved, str)
        assert len(improved) > 0
    
    @pytest.mark.asyncio
    async def test_execute_code(self, agent):
        """Test code execution."""
        # Mock subprocess to avoid actually running code
        with patch('asyncio.create_subprocess_exec') as mock_proc:
            # Configure the mock
            mock_process = MagicMock()
            mock_process.returncode = 0
            mock_process.communicate = AsyncMock(return_value=(
                json.dumps({"summary": "Test result"}).encode(),
                "".encode()
            ))
            mock_proc.return_value = mock_process
            
            result = await agent._execute_code("result = {'summary': 'Test result'}", {"data": []})
            
            assert isinstance(result, dict)
            assert "summary" in result
            assert result["summary"] == "Test result"
    
    @pytest.mark.asyncio
    async def test_analyze_financial_data(self, agent):
        """Test financial data analysis."""
        # Mock the generate and execute methods
        agent._generate_code = AsyncMock(return_value="# Test code")
        agent._execute_code = AsyncMock(return_value={"summary": "Test analysis"})
        
        result = await agent.analyze_financial_data("Test Company", [])
        
        assert isinstance(result, dict)
        assert "summary" in result
        assert result["summary"] == "Test analysis"


# Helper class for async mocks
class AsyncMock(MagicMock):
    async def __call__(self, *args, **kwargs):
        return super(AsyncMock, self).__call__(*args, **kwargs) 