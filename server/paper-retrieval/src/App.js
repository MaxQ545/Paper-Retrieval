// src/App.js
import React, { useState } from 'react';
import SearchBar from './components/SearchBar';
import PaperList from './components/PaperList';
import { fetchPapers } from './api';
import './styles/App.css';

const App = () => {
  const [papers, setPapers] = useState([]);

  const handleSearch = async (query) => {
    const results = await fetchPapers(query);
    setPapers(results);
  };

  return (
    <div className="app">
      <h1>Paper Retrieval</h1>
      <SearchBar onSearch={handleSearch} />
      <PaperList papers={papers} />
    </div>
  );
};

export default App;
